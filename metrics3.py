import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from rl_env1 import DynamicPricingEnv
from rl_env_sac import DynamicPricingEnvSAC
from rl_train_dqn import QNetwork as DQNNet
from rl_train_ddqn import QNetwork as DDQNNet
from rl_train_sac import Actor

# =====================================================
# CONFIG
# =====================================================
GAMMA = 0.99

price_actions = [200, 300, 400, 500, 600, 700, 800]

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(
    "master_dynamic_pricing_dataset.csv",
    parse_dates=["order_date"]
)
df["order_date_only"] = df["order_date"].dt.date

# =====================================================
# POLICY RUNNERS
# =====================================================
def run_policy_discrete(env, model=None):
    state = env.reset()
    done = False
    log = defaultdict(list)

    while not done:
        if model is None:
            avg_price = env.daily.iloc[env.idx]["avg_price"]
            action = np.argmin(np.abs(np.array(env.price_actions) - avg_price))
        else:
            with torch.no_grad():
                action = model(torch.FloatTensor(state)).argmax().item()

        price = env.price_actions[action]
        next_state, reward, done, info = env.step(action)

        log["price"].append(price)
        log["reward"].append(reward)
        log["revenue"].append(info["revenue"])
        log["sold"].append(info["sold"])

        state = next_state

    return log


def run_policy_sac(env, actor):
    state = env.reset()
    done = False
    log = defaultdict(list)

    while not done:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            mu, _ = actor(state_t)
            price = torch.clamp(200 + (mu + 1) * 300, 200, 800).item()

        next_state, reward, done, info = env.step(price)

        log["price"].append(price)
        log["reward"].append(reward)
        log["revenue"].append(info["revenue"])
        log["sold"].append(info["sold"])

        state = next_state

    return log

# =====================================================
# METRIC HELPERS
# =====================================================
def discounted_return(rewards, gamma=0.99):
    return sum((gamma ** t) * r for t, r in enumerate(rewards))


def sharpe_ratio(rewards):
    std = np.std(rewards)
    return np.mean(rewards) / std if std > 0 else 0.0


def action_entropy(prices):
    _, counts = np.unique(prices, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-9))


def price_change_frequency(prices):
    return np.mean(np.diff(prices) != 0)


def regret_vs_best_price(log):
    prices = np.array(log["price"])
    sold = np.array(log["sold"])
    if len(prices) == 0:
        return 0.0
    best_price = prices[np.argmax(prices * sold)]
    best_revenue = best_price * sold.sum()
    return best_revenue - np.sum(log["revenue"])


def compute_metrics(log, gamma=0.99):
    rewards = np.array(log["reward"])
    prices = np.array(log["price"])
    revenue = np.array(log["revenue"])
    sold = np.array(log["sold"])

    return {
        # Reward metrics
        "total_reward": rewards.sum(),
        "avg_reward": rewards.mean(),
        "reward_std": rewards.std(),
        "discounted_return": discounted_return(rewards, gamma),
        "reward_sharpe": sharpe_ratio(rewards),

        # Business metrics
        "total_revenue": revenue.sum(),
        "revenue_per_step": revenue.mean(),
        "units_sold": sold.sum(),
        "units_per_step": sold.mean(),

        # Policy behavior
        "avg_price": prices.mean(),
        "price_volatility": prices.std(),
        "price_change_freq": price_change_frequency(prices),
        "action_entropy": action_entropy(prices),

        # Efficiency
        "episode_length": len(rewards),

        # Optimality
        "regret_vs_best_price": regret_vs_best_price(log),
    }


def price_direction(prices):
    return [
        1 if prices[i] > prices[i - 1]
        else -1 if prices[i] < prices[i - 1]
        else 0
        for i in range(1, len(prices))
    ]


def proxy_metrics(name, y_true, y_pred):
    m = min(len(y_true), len(y_pred))
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_true[:m], y_pred[:m]))
    print("Precision:", precision_score(y_true[:m], y_pred[:m], average="macro", zero_division=0))
    print("Recall   :", recall_score(y_true[:m], y_pred[:m], average="macro", zero_division=0))
    print("F1 Score :", f1_score(y_true[:m], y_pred[:m], average="macro", zero_division=0))


def print_metrics(title, metrics, baseline=None):
    print(f"\n================ {title} =================")
    for k, v in metrics.items():
        if baseline and k in baseline:
            diff = v - baseline[k]
            sign = "+" if diff >= 0 else ""
            print(f"{k:25s}: {v:10.2f} ({sign}{diff:.2f})")
        else:
            print(f"{k:25s}: {v:10.2f}")

# =====================================================
# BASELINE
# =====================================================
env = DynamicPricingEnv(df, price_actions)
baseline_log = run_policy_discrete(env)
baseline_metrics = compute_metrics(baseline_log)
baseline_dir = price_direction(baseline_log["price"])

print_metrics("BASELINE", baseline_metrics)

# =====================================================
# DQN
# =====================================================
env = DynamicPricingEnv(df, price_actions)
dqn = DQNNet(len(env.reset()), len(price_actions))
dqn.load_state_dict(torch.load("models/dqn_pricing_model.pt"))
dqn.eval()

dqn_log = run_policy_discrete(env, dqn)
dqn_metrics = compute_metrics(dqn_log)
dqn_dir = price_direction(dqn_log["price"])

print_metrics("DQN", dqn_metrics, baseline_metrics)

# =====================================================
# DDQN
# =====================================================
env = DynamicPricingEnv(df, price_actions)
ddqn = DDQNNet(len(env.reset()), len(price_actions))
ddqn.load_state_dict(torch.load("models/ddqn_pricing_model.pt"))
ddqn.eval()

ddqn_log = run_policy_discrete(env, ddqn)
ddqn_metrics = compute_metrics(ddqn_log)
ddqn_dir = price_direction(ddqn_log["price"])

print_metrics("DDQN", ddqn_metrics, baseline_metrics)

# =====================================================
# SAC
# =====================================================
env_sac = DynamicPricingEnvSAC(df)
actor = Actor()
actor.load_state_dict(torch.load("models/sac_pricing_actor.pt"))
actor.eval()

sac_log = run_policy_sac(env_sac, actor)
sac_metrics = compute_metrics(sac_log)
sac_dir = price_direction(sac_log["price"])

print_metrics("SAC", sac_metrics, baseline_metrics)

# =====================================================
# PROXY METRICS
# =====================================================
print("\n========== PROXY METRICS ==========")
proxy_metrics("DQN vs Baseline", baseline_dir, dqn_dir)
proxy_metrics("DDQN vs Baseline", baseline_dir, ddqn_dir)
proxy_metrics("SAC vs Baseline", baseline_dir, sac_dir)

# =====================================================
# REVENUE UPLIFT
# =====================================================
print("\n========== REVENUE UPLIFT ==========")
print(f"DQN  : {(dqn_metrics['total_revenue'] - baseline_metrics['total_revenue']) / baseline_metrics['total_revenue'] * 100:+.2f}%")
print(f"DDQN : {(ddqn_metrics['total_revenue'] - baseline_metrics['total_revenue']) / baseline_metrics['total_revenue'] * 100:+.2f}%")
print(f"SAC  : {(sac_metrics['total_revenue'] - baseline_metrics['total_revenue']) / baseline_metrics['total_revenue'] * 100:+.2f}%")
