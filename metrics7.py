# =====================================================
# evaluation_metrics.py
# =====================================================

import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from rl_env1 import DynamicPricingEnv
from rl_env_sac import DynamicPricingEnvSAC
from rl_train_dqn import QNetwork as DQNNet
from rl_train_ddqn import QNetwork as DDQNNet
from rl_train_sac import Actor

# =====================================================
# CONFIG
# =====================================================
PRICE_ACTIONS = np.array([200, 300, 400, 500, 600, 700, 800])
GAMMA = 0.99
MIN_METRIC_VALUE = 1.0   # <<< ensures [1, ∞)

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

        state = next_state

    return log


def run_policy_sac(env, actor):
    state = env.reset()
    done = False
    log = defaultdict(list)

    while not done:
        with torch.no_grad():
            mu, _ = actor(torch.FloatTensor(state).unsqueeze(0))
            price = torch.clamp(
                200 + (mu + 1) * 300,
                200,
                800
            ).item()

        next_state, reward, done, info = env.step(price)

        log["price"].append(price)
        log["reward"].append(reward)
        log["revenue"].append(info["revenue"])
        log["sold"].append(info["sold"])

        state = next_state

    return log

# =====================================================
# METRICS
# =====================================================
def compute_metrics(log, baseline_avg_price=None):
    prices = np.array(log["price"])
    rewards = np.array(log["reward"])
    revenue = np.array(log["revenue"])
    sold = np.array(log["sold"])

    # ---- Price volatility (≥ 1)
    price_volatility = max(prices.std(), 1.0)

    # ---- Price movement intensity (≥ 1%)
    price_deltas = np.abs(np.diff(prices))
    if len(price_deltas) > 0:
        price_change_freq = price_deltas.mean() / max(prices.mean(), 1.0)
    else:
        price_change_freq = 0.0

    price_change_freq = max(price_change_freq, 0.01)

    # ---- Regret vs baseline (≥ 1)
    if baseline_avg_price is not None:
        oracle_revenue = baseline_avg_price * sold.sum()
        regret = oracle_revenue - revenue.sum()
        regret = max(regret, 1.0)
    else:
        regret = 1.0

    return {
        "total_revenue": revenue.sum(),
        "avg_reward": rewards.mean(),
        "reward_std": rewards.std(),
        "discounted_return": np.sum(
            rewards * (GAMMA ** np.arange(len(rewards)))
        ),
        "units_sold": sold.sum(),
        "avg_price": prices.mean(),
        "price_volatility": price_volatility,
        "price_change_freq": price_change_freq,
        "episode_length": len(prices),
        "regret_vs_best_price": regret
    }



# =====================================================
# MAIN EVALUATION ENTRY POINT
# =====================================================
def evaluate_all_models(df):
    results = {}

    # ---------------- BASELINE ----------------
    env = DynamicPricingEnv(df, PRICE_ACTIONS)
    baseline_log = run_policy_discrete(env)
    baseline_metrics = compute_metrics(baseline_log)

    baseline_avg_price = baseline_metrics["avg_price"]
    results["Baseline"] = baseline_metrics

    # ---------------- DQN ----------------
    env = DynamicPricingEnv(df, PRICE_ACTIONS)
    dqn = DQNNet(len(env.reset()), len(PRICE_ACTIONS))
    dqn.load_state_dict(torch.load("models/dqn_pricing_model.pt"))
    dqn.eval()

    dqn_log = run_policy_discrete(env, dqn)
    results["DQN"] = compute_metrics(
        dqn_log,
        baseline_avg_price
    )

    # ---------------- DDQN ----------------
    env = DynamicPricingEnv(df, PRICE_ACTIONS)
    ddqn = DDQNNet(len(env.reset()), len(PRICE_ACTIONS))
    ddqn.load_state_dict(torch.load("models/ddqn_pricing_model.pt"))
    ddqn.eval()

    ddqn_log = run_policy_discrete(env, ddqn)
    results["DDQN"] = compute_metrics(
        ddqn_log,
        baseline_avg_price
    )

    # ---------------- SAC ----------------
    env_sac = DynamicPricingEnvSAC(df)
    actor = Actor()
    actor.load_state_dict(torch.load("models/sac_pricing_actor.pt"))
    actor.eval()

    sac_log = run_policy_sac(env_sac, actor)
    results["SAC"] = compute_metrics(
        sac_log,
        baseline_avg_price
    )

    return results
