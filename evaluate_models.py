import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from rl_env1 import DynamicPricingEnv
from rl_train1 import QNetwork

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(
    "master_dynamic_pricing_dataset.csv",
    parse_dates=["order_date"]
)
df["order_date_only"] = df["order_date"].dt.date

price_actions = [200, 300, 400, 500, 600, 700, 800]

# ---------------------------
# LOAD MODELS
# ---------------------------
models = {
    "RL_v1_inventory": "models/ddqn_pricing_model.pt",   # inventory-penalized
    "RL_v2_revenue": "models/dqn_pricing_model.pt"       # revenue-only
}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def run_policy(env, model=None):
    """
    Runs a pricing policy and returns logs
    """
    state = env.reset()
    done = False

    log = defaultdict(list)

    while not done:
        if model is None:
            # Baseline: use average historical price
            action = np.argmin(
                np.abs(env.price_actions - env.daily.iloc[env.idx]["avg_price"])
            )
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

def compute_metrics(log):
    metrics = {}
    metrics["total_revenue"] = np.sum(log["revenue"])
    metrics["avg_reward"] = np.mean(log["reward"])
    metrics["reward_std"] = np.std(log["reward"])
    metrics["avg_price"] = np.mean(log["price"])
    metrics["price_volatility"] = np.std(log["price"])
    metrics["units_sold"] = np.sum(log["sold"])
    return metrics

# ---------------------------
# BASELINE EVALUATION
# ---------------------------
print("\n=== BASELINE PRICING ===")
env = DynamicPricingEnv(df, price_actions)
baseline_log = run_policy(env, model=None)
baseline_metrics = compute_metrics(baseline_log)

for k, v in baseline_metrics.items():
    print(f"{k:20s}: {v:.2f}")

# ---------------------------
# RL MODELS EVALUATION
# ---------------------------
results = {}

for name, path in models.items():
    print(f"\n=== {name.upper()} ===")

    env = DynamicPricingEnv(df, price_actions)
    state_dim = len(env.reset())
    action_dim = len(price_actions)

    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(path))
    model.eval()

    log = run_policy(env, model)
    metrics = compute_metrics(log)
    results[name] = metrics

    for k, v in metrics.items():
        print(f"{k:20s}: {v:.2f}")

# ---------------------------
# REVENUE UPLIFT
# ---------------------------
print("\n=== REVENUE UPLIFT vs BASELINE ===")
for name, m in results.items():
    uplift = (
        m["total_revenue"] - baseline_metrics["total_revenue"]
    ) / baseline_metrics["total_revenue"]

    print(f"{name:20s}: {uplift*100:+.2f}%")
