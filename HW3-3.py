# ============================================
# HW3-3
# Enhanced DQN Random Mode
# PyTorch Lightning Version
#
# Features
# 1. Random Mode GridWorld
# 2. Experience Replay
# 3. Target Network
# 4. Double DQN
# 5. Gradient Clipping
# 6. LR Scheduler
# 7. Save Loss / Reward Figure
# ============================================

# ============================================
# Imports
# ============================================

import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# ============================================
# Import Environment
# ============================================

from Gridworld import Gridworld

# ============================================
# Output Folder
# ============================================

output_dir = "HW3-3_output"

os.makedirs(output_dir, exist_ok=True)

# ============================================
# Random Seed
# ============================================

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ============================================
# DQN Network
# ============================================

class DQN(nn.Module):

    def __init__(self):

        super(DQN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(64, 150),
            nn.ReLU(),

            nn.Linear(150, 100),
            nn.ReLU(),

            nn.Linear(100, 4)

        )

    def forward(self, x):

        return self.network(x)

# ============================================
# Lightning DQN
# ============================================

class LightningDQN(pl.LightningModule):

    def __init__(self):

        super().__init__()

        # ====================================
        # Main Model
        # ====================================

        self.model = DQN()

        # ====================================
        # Target Model
        # ====================================

        self.target_model = copy.deepcopy(
            self.model
        )

        # ====================================
        # Replay Buffer
        # ====================================

        self.replay = deque(maxlen=1000)

        # ====================================
        # Hyperparameters
        # ====================================

        self.gamma = 0.9

        self.epsilon = 0.3

        self.batch_size = 200

        self.max_moves = 50

        self.sync_freq = 500

        # ====================================
        # Loss Function
        # ====================================

        self.loss_fn = nn.MSELoss()

        # ====================================
        # Statistics
        # ====================================

        self.losses = []

        self.rewards = []

        self.win_history = []

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

        # ====================================
        # Manual Optimization
        # ====================================

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

            lr=1e-3

        )

        # ====================================
        # Learning Rate Scheduler
        # ====================================

        scheduler = optim.lr_scheduler.ExponentialLR(

            optimizer,

            gamma=0.999

        )

        return {

            "optimizer": optimizer,

            "lr_scheduler": scheduler

        }

    # ========================================
    # Play One Episode
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
            # Q Prediction
            # =================================

            q_values = self.model(state)

            # =================================
            # Epsilon Greedy
            # =================================

            if random.random() < self.epsilon:

                action_idx = np.random.randint(0,4)

            else:

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

            done_flag = 1.0 if reward > 0 else 0.0

            # =================================
            # Store Replay
            # =================================

            self.replay.append(

                (

                    state,

                    action_idx,

                    reward,

                    next_state,

                    done_flag

                )

            )

            state = next_state

            # =================================
            # Replay Training
            # =================================

            if len(self.replay) > self.batch_size:

                minibatch = random.sample(

                    self.replay,

                    self.batch_size

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

                # =============================
                # Current Q
                # =============================

                current_q = self.model(states)

                current_q = current_q.gather(

                    1,

                    actions.unsqueeze(1)

                ).squeeze()

                # =============================
                # Double DQN Target
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

                        self.gamma
                        * (1 - dones)
                        * next_q

                    )

                # =============================
                # Loss
                # =============================

                loss = self.loss_fn(

                    current_q,

                    target_q

                )

                optimizer = self.optimizers()

                optimizer.zero_grad()

                self.manual_backward(loss)

                # =============================
                # Gradient Clipping
                # =============================

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    max_norm=1.0

                )

                optimizer.step()

                scheduler = self.lr_schedulers()

                scheduler.step()

                self.losses.append(
                    loss.item()
                )

            # =================================
            # Sync Target Network
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

        return 500

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

model = LightningDQN()

# ============================================
# Trainer
# ============================================

trainer = pl.Trainer(

    max_epochs=1,

    enable_checkpointing=False,

    logger=False

)

# ============================================
# Training
# ============================================

print("\nTraining Start...\n")

trainer.fit(

    model,

    train_loader

)

print("\nTraining Finished.\n")

# ============================================
# Save Loss Figure
# ============================================

plt.figure(figsize=(10,6))

plt.plot(model.losses)

plt.title("Training Loss")

plt.xlabel("Training Step")

plt.ylabel("Loss")

plt.grid()

plt.savefig(

    os.path.join(

        output_dir,

        "loss_curve.png"

    )

)

plt.show()

# ============================================
# Save Reward Figure
# ============================================

plt.figure(figsize=(10,6))

plt.plot(model.rewards)

plt.title("Episode Reward")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid()

plt.savefig(

    os.path.join(

        output_dir,

        "reward_curve.png"

    )

)

plt.show()

# ============================================
# Moving Average Reward
# ============================================

def moving_average(data, window=20):

    return np.convolve(

        data,

        np.ones(window)/window,

        mode='valid'

    )

plt.figure(figsize=(10,6))

plt.plot(

    moving_average(model.rewards),

    label='Moving Average Reward'

)

plt.title("Moving Average Reward")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.legend()

plt.grid()

plt.savefig(

    os.path.join(

        output_dir,

        "moving_avg_reward.png"

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

        "lightning_dqn_random.pth"

    )

)

print("\nModel Saved.\n")

# ============================================
# Testing Function
# ============================================

def test_model(model):

    game = Gridworld(

        size=4,

        mode='random'

    )

    state = game.board.render_np()

    state = state.reshape(1,64)

    state = torch.FloatTensor(state)

    print("\nInitial Board:\n")

    print(game.display())

    for step in range(20):

        q_values = model(state)

        action_idx = torch.argmax(
            q_values
        ).item()

        action = model.action_set[action_idx]

        print(f"\nStep {step+1}")

        print("Action:", action)

        game.makeMove(action)

        print(game.display())

        reward = game.reward()

        if reward == 10:

            print("\nGame Won!")

            break

        elif reward == -10:

            print("\nGame Lost!")

            break

        next_state = game.board.render_np()

        next_state = next_state.reshape(1,64)

        state = torch.FloatTensor(
            next_state
        )

# ============================================
# Run Testing
# ============================================

print("\nTesting Model...\n")

test_model(model)

# ============================================
# Win Rate Test
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

            q_values = model(state)

            action_idx = torch.argmax(
                q_values
            ).item()

            action = model.action_set[action_idx]

            game.makeMove(action)

            reward = game.reward()

            next_state = game.board.render_np()

            next_state = next_state.reshape(1,64)

            state = torch.FloatTensor(next_state)

            moves += 1

            if reward == 10:

                wins += 1

                done = True

            elif reward == -10:

                done = True

            if moves > 20:

                done = True

    win_rate = wins / num_games

    print(f"\nWin Rate: {win_rate*100:.2f}%")

# ============================================
# Evaluate
# ============================================

evaluate_model(model)