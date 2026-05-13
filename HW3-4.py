# ============================================
# HW3-4 Bonus
# Rainbow DQN for Random Mode GridWorld
#
# Features
# 1. Double DQN
# 2. Dueling Network
# 3. Prioritized Experience Replay
# 4. Multi-Step Learning
# 5. Noisy Network
# 6. Target Network
# 7. Gradient Clipping
# 8. Learning Rate Scheduler
# 9. PyTorch Lightning
# ============================================

# ============================================
# Imports
# ============================================

import os
import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

# ============================================
# Import Environment
# ============================================

from Gridworld import Gridworld

# ============================================
# Output Folder
# ============================================

output_dir = "HW3-4_output"

os.makedirs(output_dir, exist_ok=True)

# ============================================
# Random Seed
# ============================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ============================================
# Noisy Linear Layer
# ============================================

class NoisyLinear(nn.Module):

    def __init__(

        self,
        in_features,
        out_features,
        sigma_init=0.017

    ):

        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Mean Parameters
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.register_buffer(

            "weight_epsilon",

            torch.empty(out_features, in_features)

        )

        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )

        self.bias_sigma = nn.Parameter(
            torch.empty(out_features)
        )

        self.register_buffer(

            "bias_epsilon",

            torch.empty(out_features)

        )

        self.reset_parameters(
            sigma_init
        )

        self.reset_noise()

    def reset_parameters(self, sigma_init):

        mu_range = 1 / math.sqrt(
            self.in_features
        )

        self.weight_mu.data.uniform_(
            -mu_range,
            mu_range
        )

        self.weight_sigma.data.fill_(
            sigma_init
        )

        self.bias_mu.data.uniform_(
            -mu_range,
            mu_range
        )

        self.bias_sigma.data.fill_(
            sigma_init
        )

    def reset_noise(self):

        self.weight_epsilon.normal_()

        self.bias_epsilon.normal_()

    def forward(self, x):

        if self.training:

            weight = self.weight_mu + (

                self.weight_sigma
                * self.weight_epsilon

            )

            bias = self.bias_mu + (

                self.bias_sigma
                * self.bias_epsilon

            )

        else:

            weight = self.weight_mu

            bias = self.bias_mu

        return F.linear(x, weight, bias)

# ============================================
# Prioritized Replay Buffer
# ============================================

class PrioritizedReplayBuffer:

    def __init__(

        self,
        capacity,
        alpha=0.6

    ):

        self.capacity = capacity

        self.alpha = alpha

        self.buffer = []

        self.priorities = np.zeros(
            (capacity,),
            dtype=np.float32
        )

        self.position = 0

    def push(

        self,
        state,
        action,
        reward,
        next_state,
        done

    ):

        max_priority = (

            self.priorities.max()

            if self.buffer

            else 1.0

        )

        if len(self.buffer) < self.capacity:

            self.buffer.append(

                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

            )

        else:

            self.buffer[self.position] = (

                state,
                action,
                reward,
                next_state,
                done

            )

        self.priorities[
            self.position
        ] = max_priority

        self.position = (

            self.position + 1
        ) % self.capacity

    def sample(

        self,
        batch_size,
        beta=0.4

    ):

        if len(self.buffer) == self.capacity:

            priorities = self.priorities

        else:

            priorities = self.priorities[
                :len(self.buffer)
            ]

        probabilities = (
            priorities ** self.alpha
        )

        probabilities /= probabilities.sum()

        indices = np.random.choice(

            len(self.buffer),

            batch_size,

            p=probabilities

        )

        samples = [

            self.buffer[idx]
            for idx in indices

        ]

        total = len(self.buffer)

        weights = (

            total
            * probabilities[indices]

        ) ** (-beta)

        weights /= weights.max()

        weights = np.array(
            weights,
            dtype=np.float32
        )

        return (

            samples,
            indices,
            weights

        )

    def update_priorities(

        self,
        indices,
        priorities

    ):

        for idx, priority in zip(

            indices,
            priorities

        ):

            self.priorities[idx] = (
                priority + 1e-5
            )

    def __len__(self):

        return len(self.buffer)

# ============================================
# Rainbow DQN Network
# Dueling + Noisy
# ============================================

class RainbowDQN(nn.Module):

    def __init__(self):

        super(RainbowDQN, self).__init__()

        self.feature = nn.Sequential(

            nn.Linear(64, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU()

        )

        # Value Stream

        self.value_stream = nn.Sequential(

            NoisyLinear(128, 64),
            nn.ReLU(),

            NoisyLinear(64, 1)

        )

        # Advantage Stream

        self.advantage_stream = nn.Sequential(

            NoisyLinear(128, 64),
            nn.ReLU(),

            NoisyLinear(64, 4)

        )

    def forward(self, x):

        features = self.feature(x)

        value = self.value_stream(
            features
        )

        advantage = self.advantage_stream(
            features
        )

        q_values = value + (

            advantage
            - advantage.mean(dim=1, keepdim=True)

        )

        return q_values

    def reset_noise(self):

        for module in self.modules():

            if isinstance(module, NoisyLinear):

                module.reset_noise()

# ============================================
# Lightning Rainbow DQN
# ============================================

class LightningRainbowDQN(

    pl.LightningModule

):

    def __init__(self):

        super().__init__()

        # ====================================
        # Main Network
        # ====================================

        self.model = RainbowDQN()

        # ====================================
        # Target Network
        # ====================================

        self.target_model = copy.deepcopy(
            self.model
        )

        # ====================================
        # Replay Buffer
        # ====================================

        self.replay = PrioritizedReplayBuffer(
            capacity=5000
        )

        # ====================================
        # Hyperparameters
        # ====================================

        self.gamma = 0.99

        self.batch_size = 64

        self.sync_freq = 200

        self.max_moves = 50

        self.multi_step = 3

        self.beta = 0.4

        self.beta_increment = 1e-4

        # ====================================
        # Multi Step Buffer
        # ====================================

        self.n_step_buffer = deque(
            maxlen=self.multi_step
        )

        # ====================================
        # Statistics
        # ====================================

        self.losses = []

        self.rewards = []

        self.step_counter = 0

        # ====================================
        # Action Mapping
        # ====================================

        self.action_set = {

            0: 'u',
            1: 'd',
            2: 'l',
            3: 'r'

        }

        self.automatic_optimization = False

    # ========================================
    # Forward
    # ========================================

    def forward(self, x):

        return self.model(x)

    # ========================================
    # Optimizer
    # ========================================

    def configure_optimizers(self):

        optimizer = optim.Adam(

            self.model.parameters(),

            lr=1e-4

        )

        scheduler = optim.lr_scheduler.StepLR(

            optimizer,

            step_size=1000,

            gamma=0.5

        )

        return {

            "optimizer": optimizer,

            "lr_scheduler": scheduler

        }

    # ========================================
    # Multi Step Return
    # ========================================

    def get_n_step_info(self):

        reward, next_state, done = (

            self.n_step_buffer[-1][2],
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4]

        )

        for transition in reversed(

            list(self.n_step_buffer)[:-1]

        ):

            r, n_s, d = (

                transition[2],
                transition[3],
                transition[4]

            )

            reward = r + (

                self.gamma
                * reward
                * (1 - d)

            )

            next_state, done = (

                n_s,
                d

            )

        return (

            self.n_step_buffer[0][0],
            self.n_step_buffer[0][1],
            reward,
            next_state,
            done

        )

    # ========================================
    # Play Episode
    # ========================================

    def play_episode(self):

        game = Gridworld(

            size=4,

            mode='random'

        )

        state = game.board.render_np()

        state = state.reshape(1,64)

        state = state + np.random.rand(1,64)/100.0

        state = torch.FloatTensor(state)

        done = False

        total_reward = 0

        moves = 0

        while not done:

            self.step_counter += 1

            moves += 1

            # =================================
            # Noisy Reset
            # =================================

            self.model.reset_noise()

            self.target_model.reset_noise()

            # =================================
            # Action
            # =================================

            with torch.no_grad():

                q_values = self.model(state)

            action_idx = torch.argmax(
                q_values
            ).item()

            action = self.action_set[action_idx]

            # =================================
            # Environment Step
            # =================================

            game.makeMove(action)

            next_state = game.board.render_np()

            next_state = next_state.reshape(1,64)

            next_state = (

                next_state

                + np.random.rand(1,64)/100.0

            )

            next_state = torch.FloatTensor(
                next_state
            )

            reward = game.reward()

            total_reward += reward

            done_flag = (

                1.0 if reward != -1
                else 0.0

            )

            # =================================
            # N-Step Buffer
            # =================================

            self.n_step_buffer.append(

                (
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done_flag
                )

            )

            if len(self.n_step_buffer) == self.multi_step:

                transition = self.get_n_step_info()

                self.replay.push(*transition)

            state = next_state

            # =================================
            # Training
            # =================================

            if len(self.replay) > self.batch_size:

                (
                    minibatch,
                    indices,
                    weights

                ) = self.replay.sample(

                    self.batch_size,
                    beta=self.beta

                )

                states = torch.cat([

                    s1 for (
                        s1,a,r,s2,d
                    ) in minibatch

                ])

                actions = torch.LongTensor([

                    a for (
                        s1,a,r,s2,d
                    ) in minibatch

                ])

                rewards = torch.FloatTensor([

                    r for (
                        s1,a,r,s2,d
                    ) in minibatch

                ])

                next_states = torch.cat([

                    s2 for (
                        s1,a,r,s2,d
                    ) in minibatch

                ])

                dones = torch.FloatTensor([

                    d for (
                        s1,a,r,s2,d
                    ) in minibatch

                ])

                weights = torch.FloatTensor(
                    weights
                )

                # =============================
                # Current Q
                # =============================

                current_q = self.model(states)

                current_q = current_q.gather(

                    1,

                    actions.unsqueeze(1)

                ).squeeze()

                # =============================
                # Double DQN
                # =============================

                with torch.no_grad():

                    next_actions = torch.argmax(

                        self.model(next_states),

                        dim=1

                    )

                    next_q_target = self.target_model(
                        next_states
                    )

                    next_q = next_q_target.gather(

                        1,

                        next_actions.unsqueeze(1)

                    ).squeeze()

                    target_q = rewards + (

                        (self.gamma ** self.multi_step)

                        * (1 - dones)

                        * next_q

                    )

                # =============================
                # PER Weighted Loss
                # =============================

                td_error = target_q - current_q

                loss = (

                    weights
                    * td_error.pow(2)

                ).mean()

                optimizer = self.optimizers()

                optimizer.zero_grad()

                self.manual_backward(loss)

                # =============================
                # Gradient Clipping
                # =============================

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    max_norm=10.0

                )

                optimizer.step()

                scheduler = self.lr_schedulers()

                scheduler.step()

                # =============================
                # Update Priorities
                # =============================

                priorities = (

                    td_error.abs()
                    .detach()
                    .cpu()
                    .numpy()

                )

                self.replay.update_priorities(

                    indices,
                    priorities

                )

                self.losses.append(
                    loss.item()
                )

                self.beta = min(

                    1.0,

                    self.beta + self.beta_increment

                )

            # =================================
            # Sync Target
            # =================================

            if self.step_counter % self.sync_freq == 0:

                self.target_model.load_state_dict(

                    self.model.state_dict()

                )

            # =================================
            # Terminal
            # =================================

            if reward != -1:

                done = True

            if moves > self.max_moves:

                done = True

        self.rewards.append(total_reward)

        return total_reward

    # ========================================
    # Training Step
    # ========================================

    def training_step(

        self,
        batch,
        batch_idx

    ):

        reward = self.play_episode()

        self.log(

            "episode_reward",

            reward

        )

        return torch.tensor(
            reward,
            dtype=torch.float32
        )

# ============================================
# Dummy Dataset
# ============================================

class DummyDataset(

    torch.utils.data.Dataset

):

    def __len__(self):

        return 1000

    def __getitem__(self, idx):

        return idx

# ============================================
# DataLoader
# ============================================

train_loader = torch.utils.data.DataLoader(

    DummyDataset(),

    batch_size=1

)

# ============================================
# Initialize Model
# ============================================

model = LightningRainbowDQN()

# ============================================
# Trainer
# ============================================

trainer = pl.Trainer(

    max_epochs=1,

    enable_checkpointing=False,

    logger=False

)

# ============================================
# Train
# ============================================

print("\nRainbow DQN Training Start...\n")

trainer.fit(

    model,

    train_loader

)

print("\nTraining Finished.\n")

# ============================================
# Plot Loss
# ============================================

plt.figure(figsize=(10,6))

plt.plot(model.losses)

plt.title("Rainbow DQN Loss")

plt.xlabel("Training Step")

plt.ylabel("Loss")

plt.grid()

plt.savefig(

    os.path.join(

        output_dir,

        "rainbow_loss.png"

    )

)

plt.show()

# ============================================
# Plot Reward
# ============================================

plt.figure(figsize=(10,6))

plt.plot(model.rewards)

plt.title("Rainbow DQN Reward")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid()

plt.savefig(

    os.path.join(

        output_dir,

        "rainbow_reward.png"

    )

)

plt.show()

# ============================================
# Save Model
# ============================================

torch.save(

    model.model.state_dict(),

    os.path.join(

        output_dir,

        "rainbow_dqn.pth"

    )

)

print("\nRainbow DQN Model Saved.\n")

# ============================================
# Evaluation
# ============================================

def evaluate_model(

    model,
    num_games=100

):

    wins = 0

    for _ in range(num_games):

        game = Gridworld(

            size=4,

            mode='random'

        )

        state = game.board.render_np()

        state = state.reshape(1,64)

        state = torch.FloatTensor(state)

        done = False

        moves = 0

        while not done:

            with torch.no_grad():

                q_values = model(state)

            action_idx = torch.argmax(
                q_values
            ).item()

            action = model.action_set[action_idx]

            game.makeMove(action)

            reward = game.reward()

            next_state = game.board.render_np()

            next_state = next_state.reshape(1,64)

            state = torch.FloatTensor(
                next_state
            )

            moves += 1

            if reward == 10:

                wins += 1

                done = True

            elif reward == -10:

                done = True

            if moves > 20:

                done = True

    win_rate = wins / num_games

    print(f"\nRainbow DQN Win Rate: {win_rate*100:.2f}%")

# ============================================
# Evaluate
# ============================================

evaluate_model(model)