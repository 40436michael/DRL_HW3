# ============================================
# HW3-1 Naive DQN + Experience Replay
# GridWorld Reinforcement Learning
# ============================================

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
import os
# ============================================
# Utility Functions
# ============================================
output_dir = "HW3-1_output"

os.makedirs(output_dir, exist_ok=True)
def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)

def addTuple(a, b):
    return tuple([sum(x) for x in zip(a, b)])

# ============================================
# GridBoard Classes
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
# GridWorld Environment
# ============================================

class Gridworld:

    def __init__(self, size=4, mode='static'):

        self.size = size
        self.board = GridBoard(size=size)

        self.board.addPiece('Player', 'P')
        self.board.addPiece('Goal', '+')
        self.board.addPiece('Pit', '-')
        self.board.addPiece('Wall', 'W')

        if mode == 'static':
            self.initGridStatic()

        elif mode == 'player':
            self.initGridPlayer()

        else:
            self.initGridRand()

    # ========================================
    # Static Mode
    # ========================================

    def initGridStatic(self):

        self.board.components['Player'].pos = (0,3)
        self.board.components['Goal'].pos = (0,0)
        self.board.components['Pit'].pos = (0,1)
        self.board.components['Wall'].pos = (1,1)

    # ========================================
    # Player Mode
    # ========================================

    def initGridPlayer(self):

        self.initGridStatic()

        self.board.components['Player'].pos = randPair(0, self.size)

        if not self.validateBoard():
            self.initGridPlayer()

    # ========================================
    # Random Mode
    # ========================================

    def initGridRand(self):

        self.board.components['Player'].pos = randPair(0, self.size)
        self.board.components['Goal'].pos = randPair(0, self.size)
        self.board.components['Pit'].pos = randPair(0, self.size)
        self.board.components['Wall'].pos = randPair(0, self.size)

        if not self.validateBoard():
            self.initGridRand()

    # ========================================
    # Validate Board
    # ========================================

    def validateBoard(self):

        positions = []

        for piece in self.board.components.values():
            positions.append(piece.pos)

        if len(positions) != len(set(positions)):
            return False

        return True

    # ========================================
    # Move Validation
    # ========================================

    def validateMove(self, piece, move):

        pos = addTuple(self.board.components[piece].pos, move)

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

    # ========================================
    # Make Move
    # ========================================

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

    # ========================================
    # Reward Function
    # ========================================

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

    # ========================================
    # Display
    # ========================================

    def display(self):
        return self.board.render()

# ============================================
# DQN Network
# ============================================

class DQN(nn.Module):

    def __init__(self):

        super(DQN, self).__init__()

        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# ============================================
# Hyperparameters
# ============================================

gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

learning_rate = 1e-3

epochs = 3000
max_moves = 50

batch_size = 64
memory_size = 1000

# ============================================
# Initialize Model
# ============================================

model = DQN()

loss_fn = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# ============================================
# Replay Buffer
# ============================================

replay = deque(maxlen=memory_size)

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
# Training
# ============================================

losses = []
episode_rewards = []

for epoch in range(epochs):

    game = Gridworld(size=4, mode='static')

    state = game.board.render_np().reshape(1,64)
    state = state + np.random.rand(1,64)/100.0

    state = torch.FloatTensor(state)

    done = False
    moves = 0
    total_reward = 0
    while not done:

        moves += 1

        # ====================================
        # Predict Q-values
        # ====================================

        q_values = model(state)

        # ====================================
        # Epsilon-Greedy
        # ====================================

        if random.random() < epsilon:
            action_idx = np.random.randint(0,4)

        else:
            action_idx = torch.argmax(q_values).item()

        action = action_set[action_idx]

        # ====================================
        # Take Action
        # ====================================

        game.makeMove(action)

        next_state = game.board.render_np().reshape(1,64)
        next_state = next_state + np.random.rand(1,64)/100.0

        next_state = torch.FloatTensor(next_state)

        reward = game.reward()
        total_reward += reward
        if reward != -1:
            done = True

        # ====================================
        # Store Experience
        # ====================================

        replay.append(
            (state, action_idx, reward, next_state, done)
        )

        state = next_state

        # ====================================
        # Replay Training
        # ====================================

        if len(replay) >= batch_size:

            minibatch = random.sample(replay, batch_size)

            states = torch.cat([s for (s,a,r,s2,d) in minibatch])

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

            # ================================
            # Current Q
            # ================================

            current_q = model(states)

            current_q = current_q.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze()

            # ================================
            # Target Q
            # ================================

            with torch.no_grad():

                next_q = model(next_states)

                max_next_q = torch.max(
                    next_q,
                    dim=1
                )[0]

                target_q = rewards + gamma * (
                    1 - dones
                ) * max_next_q

            # ================================
            # Loss
            # ================================

            loss = loss_fn(current_q, target_q)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        if moves > max_moves:
            break

    # ========================================
    # Epsilon Decay
    # ========================================
    episode_rewards.append(total_reward)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # ========================================
    # Print Training Info
    # ========================================

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch: {epoch+1}, "
            f"Epsilon: {epsilon:.3f}, "
            f"Loss: {loss.item():.4f}"
        )

# ============================================
# Plot Loss
# ============================================

plt.figure(figsize=(10,6))

plt.plot(losses)

plt.title("DQN Training Loss")

plt.xlabel("Training Steps")

plt.ylabel("Loss")

plt.grid()

loss_path = os.path.join(
    output_dir,
    "training_loss.png"
)

plt.savefig(loss_path)

plt.show()
# ============================================
# Plot Reward Curve
# ============================================

plt.figure(figsize=(10,6))

plt.plot(episode_rewards)

plt.title("Episode Reward")

plt.xlabel("Episode")

plt.ylabel("Total Reward")

plt.grid()

reward_path = os.path.join(
    output_dir,
    "reward_curve.png"
)

plt.savefig(reward_path)

plt.show()
# ============================================
# Testing
# ============================================

def test_model(model, mode='static', save_image=True):
    game = Gridworld(size=4, mode=mode)

    state = game.board.render_np().reshape(1,64)
    state = torch.FloatTensor(state)

    print("\nInitial Board:")
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
            return

        elif reward == -10:
            print("\nGame Lost!")
            return

        next_state = game.board.render_np().reshape(1,64)

        state = torch.FloatTensor(next_state)

    print("\nToo many moves.")

# ============================================
# Run Test
# ============================================

test_model(model, mode='static')