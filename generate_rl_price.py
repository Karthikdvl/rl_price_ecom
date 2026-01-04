import pandas as pd
import torch
import numpy as np
from rl_env import DynamicPricingEnv
from rl_train import QNetwork   # same architecture

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
# LOAD TRAINED DQN / DDQN
# ---------------------------
model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load("models/ddqn_pricing_model.pt"))
model.eval()

# ---------------------------
# GENERATE RL PRICE
# ---------------------------
rl_prices = []
state = env.reset()
done = False

while not done:
    with torch.no_grad():
        action = model(torch.FloatTensor(state)).argmax().item()

    rl_price = price_actions[action]
    rl_prices.append(rl_price)

    next_state, reward, done, info = env.step(action)
    state = next_state if next_state is not None else state

# Pad if needed
df = df.iloc[:len(rl_prices)].copy()
df["rl_price"] = rl_prices

df.to_csv("data_with_rl_price.csv", index=False)

print("✅ rl_price column generated and saved")
