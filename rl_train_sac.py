import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.distributions import Normal

from rl_env_sac import DynamicPricingEnvSAC

# ---------------------------
# CONFIG
# ---------------------------
EPISODES = 300
LR = 3e-4
GAMMA = 0.99

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------
# ACTOR (SAFE TO IMPORT)
# ---------------------------
class Actor(nn.Module):
    def __init__(self, state_dim=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return mu, std

# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train_sac():
    df = pd.read_csv(
        "master_dynamic_pricing_dataset.csv",
        parse_dates=["order_date"]
    )
    df["order_date_only"] = df["order_date"].dt.date

    env = DynamicPricingEnvSAC(df)
    actor = Actor()
    optimizer = optim.Adam(actor.parameters(), lr=LR)

    reward_history = []

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        log_probs, rewards = [], []

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mu, std = actor(state_t)
            dist = Normal(mu, std)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            price = torch.clamp(200 + (action + 1) * 300, 200, 800).item()
            next_state, reward, done, _ = env.step(price)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = sum(-lp * G for lp, G in zip(log_probs, returns))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(f"[SAC] Episode {ep+1}/{EPISODES} | Reward={total_reward:.2f}")

    torch.save(actor.state_dict(), MODEL_DIR / "sac_pricing_actor.pt")
    np.save(MODEL_DIR / "sac_rewards.npy", reward_history)

    print("✅ SAC-like model training complete")

# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    train_sac()
