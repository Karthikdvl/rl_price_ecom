# rl_train.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from pathlib import Path

from rl_env1 import DynamicPricingEnv

# ---------------------------
# CONFIG
# ---------------------------
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(
    "master_dynamic_pricing_dataset.csv",
    parse_dates=["order_date"]
)
df["order_date_only"] = df["order_date"].dt.date

price_actions = [200, 300, 400, 500, 600, 700, 800]

env = DynamicPricingEnv(df, price_actions)

state_dim = len(env.reset())
action_dim = len(price_actions)

# ---------------------------
# Q-NETWORK
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---------------------------
# REPLAY BUFFER
# ---------------------------
memory = deque(maxlen=50_000)

def select_action(state, eps):
    if random.random() < eps:
        return random.randrange(action_dim)
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return policy_net(state_t).argmax().item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.from_numpy(np.array(states, dtype=np.float32))
    next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
    actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1)
    rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1)
    dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        max_next_q = target_net(next_states).gather(1, next_actions)
        target_q = rewards + GAMMA * max_next_q * (1 - dones)

    loss = loss_fn(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------
# TRAIN LOOP
# ---------------------------
eps = EPS_START
reward_history = []

for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = select_action(state, eps)
        next_state, reward, done, info = env.step(action)

        memory.append(
            (
                state,
                action,
                reward,
                next_state if next_state is not None else state,
                float(done),
            )
        )

        state = next_state if next_state is not None else state
        total_reward += reward

        optimize_model()

    eps = max(EPS_END, eps * EPS_DECAY)
    reward_history.append(total_reward)

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (ep + 1) % 50 == 0:
        print(
            f"Episode {ep+1}/{EPISODES} | "
            f"Reward={total_reward:.2f} | eps={eps:.3f}"
        )

# ---------------------------
# SAVE OUTPUTS
# ---------------------------
torch.save(policy_net.state_dict(), MODEL_DIR / "dqn_pricing_model.pt")
np.save(MODEL_DIR / "dqn_rewards.npy", reward_history)

print("✅ DQN training complete. Model and rewards saved.")
