import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from rl_env1 import DynamicPricingEnv
from rl_env_sac import DynamicPricingEnvSAC
from rl_train2_dqn import QNetwork
from rl_train_sac import Actor

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(
    "master_dynamic_pricing_dataset.csv",
    parse_dates=["order_date"]
)
df["order_date_only"] = df["order_date"].dt.date

price_actions = [200, 300, 400, 500, 600, 700, 800]

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def run_policy_discrete(env, model=None):
    state = env.reset()
    done = False
    log = defaultdict(list)

    while not done:
        if model is None:
            avg_price = env.daily.iloc[env.idx]["avg_price"]
            action = np.argmin(np.abs(env.price_actions - avg_price))
        else:
            with torch.no_grad():
                action = model(torch.FloatTensor(state)).argmax().item()

        price = env.price_actions[action]
        next_state, reward, done, info = env.step(action)

        log["price"].append(price)
        log["reward"].append(reward)
        log["revenue"].append(info["revenue"])
        log["sold"].append(info["sold"])

        state = next_state if next_state is not None else state

    return log


def run_policy_sac(env, actor):
    state = env.reset()
    done = False
    log = defaultdict(list)

    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mu, std = actor(state_t)
            price = torch.clamp(
                200 + (mu + 1) * 300, 200, 800
            ).item()

        next_state, reward, done, info = env.step(price)

        log["price"].append(price)
        log["reward"].append(reward)
        log["revenue"].append(info["revenue"])
        log["sold"].append(info["sold"])

        state = next_state if next_state is not None else state

    return log


def compute_metrics(log):
    return {
        "total_revenue": np.sum(log["revenue"]),
        "avg_reward": np.mean(log["reward"]),
        "reward_std": np.std(log["reward"]),
        "avg_price": np.mean(log["price"]),
        "price_volatility": np.std(log["price"]),
        "units_sold": np.sum(log["sold"]),
    }


def price_direction(prices):
    dirs = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            dirs.append(1)
        elif prices[i] < prices[i - 1]:
            dirs.append(-1)
        else:
            dirs.append(0)
    return dirs


def print_proxy_metrics(name, y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_true[:min_len], y_pred[:min_len]))
    print(
        "Precision:",
        precision_score(y_true[:min_len], y_pred[:min_len], average="macro", zero_division=0),
    )
    print(
        "Recall   :",
        recall_score(y_true[:min_len], y_pred[:min_len], average="macro", zero_division=0),
    )
    print(
        "F1 Score :",
        f1_score(y_true[:min_len], y_pred[:min_len], average="macro", zero_division=0),
    )

# =====================================================
# BASELINE
# =====================================================
print("\n================ BASELINE =================")
env = DynamicPricingEnv(df, price_actions)
baseline_log = run_policy_discrete(env, model=None)
baseline_metrics = compute_metrics(baseline_log)

for k, v in baseline_metrics.items():
    print(f"{k:20s}: {v:.2f}")

baseline_dir = price_direction(baseline_log["price"])

# =====================================================
# DQN
# =====================================================
print("\n================ DQN =================")
env = DynamicPricingEnv(df, price_actions)
state_dim = len(env.reset())
action_dim = len(price_actions)

dqn_model = QNetwork(state_dim, action_dim)
dqn_model.load_state_dict(torch.load("models/dqn_pricing_model.pt"))
dqn_model.eval()

dqn_log = run_policy_discrete(env, dqn_model)
dqn_metrics = compute_metrics(dqn_log)

for k, v in dqn_metrics.items():
    print(f"{k:20s}: {v:.2f}")

dqn_dir = price_direction(dqn_log["price"])

# =====================================================
# DDQN
# =====================================================
print("\n================ DDQN =================")
env = DynamicPricingEnv(df, price_actions)

ddqn_model = QNetwork(state_dim, action_dim)
ddqn_model.load_state_dict(torch.load("models/ddqn_pricing_model.pt"))
ddqn_model.eval()

ddqn_log = run_policy_discrete(env, ddqn_model)
ddqn_metrics = compute_metrics(ddqn_log)

for k, v in ddqn_metrics.items():
    print(f"{k:20s}: {v:.2f}")

ddqn_dir = price_direction(ddqn_log["price"])

# =====================================================
# SAC
# =====================================================
print("\n================ SAC =================")
env_sac = DynamicPricingEnvSAC(df)

actor = Actor()
actor.load_state_dict(torch.load("models/sac_pricing_actor.pt"))
actor.eval()

sac_log = run_policy_sac(env_sac, actor)
sac_metrics = compute_metrics(sac_log)

for k, v in sac_metrics.items():
    print(f"{k:20s}: {v:.2f}")

sac_dir = price_direction(sac_log["price"])

# =====================================================
# PROXY METRICS
# =====================================================
print("\n========== PROXY CLASSIFICATION METRICS ==========")
print_proxy_metrics("DQN vs Baseline", baseline_dir, dqn_dir)
print_proxy_metrics("DDQN vs Baseline", baseline_dir, ddqn_dir)
print_proxy_metrics("SAC vs Baseline", baseline_dir, sac_dir)

# =====================================================
# REVENUE UPLIFT
# =====================================================
print("\n========== REVENUE UPLIFT vs BASELINE ==========")
print(
    f"DQN  : {(dqn_metrics['total_revenue'] - baseline_metrics['total_revenue']) / baseline_metrics['total_revenue'] * 100:+.2f}%"
)
print(
    f"DDQN : {(ddqn_metrics['total_revenue'] - baseline_metrics['total_revenue']) / baseline_metrics['total_revenue'] * 100:+.2f}%"
)
print(
    f"SAC  : {(sac_metrics['total_revenue'] - baseline_metrics['total_revenue']) / baseline_metrics['total_revenue'] * 100:+.2f}%"
)
