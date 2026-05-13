# ============================================
# HW3-2 Enhanced DQN Variants
# Double DQN + Dueling DQN
# GridWorld Player Mode
# ============================================

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
import copy
import os

# ============================================
# Output Folder
# ============================================

output_dir = "HW3-2_output"

os.makedirs(output_dir, exist_ok=True)

# ============================================
# Utility Functions
# ============================================

def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)

def addTuple(a, b):
    return tuple([sum(x) for x in zip(a, b)])

# ============================================
# GridBoard
# ============================================

class BoardPiece:

    def __init__(self, name, code, pos):
        self.name = name
        self.code = code
        self.pos = pos

class GridBoard:

    def __init__(self, size=4):

        self.size = size
        self.components = {}

    def addPiece(self, name, code, pos=(0,0)):

        piece = BoardPiece(name, code, pos)

        self.components[name] = piece

    def movePiece(self, name, pos):

        self.components[name].pos = pos

    def render(self):

        board = np.full((self.size, self.size), ' ')

        for name, piece in self.components.items():

            board[piece.pos] = piece.code

        return board

    def render_np(self):

        board = np.zeros((4, self.size, self.size))

        pieces = ['Player', 'Goal', 'Pit', 'Wall']

        for layer, name in enumerate(pieces):

            pos = self.components[name].pos

            board[layer][pos] = 1

        return board

# ============================================
# GridWorld
# ============================================

class Gridworld:

    def __init__(self, size=4, mode='player'):

        self.size = size

        self.board = GridBoard(size=size)

        self.board.addPiece('Player', 'P')
        self.board.addPiece('Goal', '+')
        self.board.addPiece('Pit', '-')
        self.board.addPiece('Wall', 'W')

        if mode == 'player':

            self.initGridPlayer()

        else:

            self.initGridStatic()

    def initGridStatic(self):

        self.board.components['Player'].pos = (0,3)
        self.board.components['Goal'].pos = (0,0)
        self.board.components['Pit'].pos = (0,1)
        self.board.components['Wall'].pos = (1,1)

    def initGridPlayer(self):

        self.initGridStatic()

        self.board.components['Player'].pos = randPair(0, self.size)

        if not self.validateBoard():

            self.initGridPlayer()

    def validateBoard(self):

        positions = []

        for piece in self.board.components.values():

            positions.append(piece.pos)

        return len(positions) == len(set(positions))

    def validateMove(self, piece, move):

        pos = addTuple(
            self.board.components[piece].pos,
            move
        )

        wall = self.board.components['Wall'].pos
        pit = self.board.components['Pit'].pos

        if pos == wall:
            return 1

        if min(pos) < 0:
            return 1

        if max(pos) > self.size - 1:
            return 1

        if pos == pit:
            return 2

        return 0

    def makeMove(self, action):

        moves = {
            'u': (-1,0),
            'd': (1,0),
            'l': (0,-1),
            'r': (0,1)
        }

        move = moves[action]

        outcome = self.validateMove('Player', move)

        if outcome in [0,2]:

            new_pos = addTuple(
                self.board.components['Player'].pos,
                move
            )

            self.board.movePiece('Player', new_pos)

    def reward(self):

        player = self.board.components['Player'].pos
        goal = self.board.components['Goal'].pos
        pit = self.board.components['Pit'].pos

        if player == goal:
            return 10

        elif player == pit:
            return -10

        else:
            return -1

    def display(self):

        return self.board.render()

# ============================================
# Double DQN Network
# ============================================

class DoubleDQN(nn.Module):

    def __init__(self):

        super(DoubleDQN, self).__init__()

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# ============================================
# Dueling DQN Network
# ============================================

class DuelingDQN(nn.Module):

    def __init__(self):

        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):

        features = self.feature(x)

        values = self.value_stream(features)

        advantages = self.advantage_stream(features)

        qvals = values + (
            advantages - advantages.mean(dim=1, keepdim=True)
        )

        return qvals

# ============================================
# Hyperparameters
# ============================================

gamma = 0.9

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

learning_rate = 1e-3

epochs = 3000

batch_size = 64

memory_size = 1000

max_moves = 50

sync_freq = 100

# ============================================
# Action Mapping
# ============================================

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r'
}

# ============================================
# Training Function
# ============================================

def train_model(model_name='double', epsilon=1.0):

    if model_name == 'double':

        model = DoubleDQN()

    else:

        model = DuelingDQN()

    target_model = copy.deepcopy(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_fn = nn.MSELoss()

    replay = deque(maxlen=memory_size)

    losses = []

    rewards_list = []

    step_count = 0

    for epoch in range(epochs):

        game = Gridworld(size=4, mode='player')

        state = game.board.render_np().reshape(1,64)

        state += np.random.rand(1,64)/100.0

        state = torch.FloatTensor(state)

        done = False

        total_reward = 0

        moves = 0

        while not done:

            moves += 1

            step_count += 1

            # ====================================
            # Action Selection
            # ====================================

            q_values = model(state)

            if random.random() < epsilon:

                action_idx = np.random.randint(0,4)

            else:

                action_idx = torch.argmax(q_values).item()

            action = action_set[action_idx]

            # ====================================
            # Environment Step
            # ====================================

            game.makeMove(action)

            next_state = game.board.render_np().reshape(1,64)

            next_state += np.random.rand(1,64)/100.0

            next_state = torch.FloatTensor(next_state)

            reward = game.reward()

            total_reward += reward

            if reward != -1:

                done = True

            # ====================================
            # Store Experience
            # ====================================

            replay.append(
                (
                    state,
                    action_idx,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state

            # ====================================
            # Replay Training
            # ====================================

            if len(replay) >= batch_size:

                minibatch = random.sample(
                    replay,
                    batch_size
                )

                states = torch.cat(
                    [s for (s,a,r,s2,d) in minibatch]
                )

                actions = torch.LongTensor(
                    [a for (s,a,r,s2,d) in minibatch]
                )

                rewards = torch.FloatTensor(
                    [r for (s,a,r,s2,d) in minibatch]
                )

                next_states = torch.cat(
                    [s2 for (s,a,r,s2,d) in minibatch]
                )

                dones = torch.FloatTensor(
                    [d for (s,a,r,s2,d) in minibatch]
                )

                # ====================================
                # Current Q
                # ====================================

                current_q = model(states)

                current_q = current_q.gather(
                    1,
                    actions.unsqueeze(1)
                ).squeeze()

                # ====================================
                # Double DQN Target
                # ====================================

                with torch.no_grad():

                    if model_name == 'double':

                        next_actions = torch.argmax(
                            model(next_states),
                            dim=1
                        )

                        next_q_target = target_model(next_states)

                        next_q = next_q_target.gather(
                            1,
                            next_actions.unsqueeze(1)
                        ).squeeze()

                    else:

                        next_q_target = target_model(next_states)

                        next_q = torch.max(
                            next_q_target,
                            dim=1
                        )[0]

                    target_q = rewards + gamma * (
                        1 - dones
                    ) * next_q

                # ====================================
                # Loss
                # ====================================

                loss = loss_fn(
                    current_q,
                    target_q
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                losses.append(loss.item())

            # ====================================
            # Sync Target Network
            # ====================================

            if step_count % sync_freq == 0:

                target_model.load_state_dict(
                    model.state_dict()
                )

            if moves > max_moves:
                done = True
                break

        rewards_list.append(total_reward)

        # ========================================
        # Epsilon Decay
        # ========================================


        if epsilon > epsilon_min:

            epsilon *= epsilon_decay

        # ========================================
        # Print Info
        # ========================================

        if (epoch + 1) % 100 == 0:

            print(
                f"{model_name.upper()} | "
                f"Epoch {epoch+1} | "
                f"Reward {total_reward} | "
                f"Loss {loss.item():.4f}"
            )

    return model, losses, rewards_list

# ============================================
# Train Double DQN
# ============================================

print("\nTraining Double DQN...\n")

double_model, double_losses, double_rewards = train_model(
    model_name='double',epsilon=1.0
)

# ============================================
# Train Dueling DQN
# ============================================

epsilon = 1.0

print("\nTraining Dueling DQN...\n")

dueling_model, dueling_losses, dueling_rewards = train_model(
    model_name='dueling', epsilon=1.0
)

# ============================================
# Plot Loss Comparison
# ============================================

plt.figure(figsize=(10,6))

plt.plot(double_losses, label='Double DQN')

plt.plot(dueling_losses, label='Dueling DQN')

plt.title("Loss Comparison")

plt.xlabel("Training Steps")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        output_dir,
        "loss_comparison.png"
    )
)

plt.show()

# ============================================
# Plot Reward Comparison
# ============================================

plt.figure(figsize=(10,6))

plt.plot(double_rewards, label='Double DQN')

plt.plot(dueling_rewards, label='Dueling DQN')

plt.title("Reward Comparison")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.legend()

plt.grid()

plt.savefig(
    os.path.join(
        output_dir,
        "reward_comparison.png"
    )
)

plt.show()

# ============================================
# Save Models
# ============================================

torch.save(
    double_model.state_dict(),
    os.path.join(
        output_dir,
        "double_dqn.pth"
    )
)

torch.save(
    dueling_model.state_dict(),
    os.path.join(
        output_dir,
        "dueling_dqn.pth"
    )
)

print("\nModels Saved.")

# ============================================
# Testing Function
# ============================================

def test_model(model, mode='player'):

    game = Gridworld(size=4, mode=mode)

    state = game.board.render_np().reshape(1,64)

    state = torch.FloatTensor(state)

    print("\nInitial Board:\n")

    print(game.display())

    for step in range(15):

        q_values = model(state)

        action_idx = torch.argmax(q_values).item()

        action = action_set[action_idx]

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

        next_state = game.board.render_np().reshape(1,64)

        state = torch.FloatTensor(next_state)

# ============================================
# Run Testing
# ============================================

print("\nTesting Double DQN")

test_model(double_model)

print("\nTesting Dueling DQN")

test_model(dueling_model)