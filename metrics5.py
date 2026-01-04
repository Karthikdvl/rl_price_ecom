# evaluation_metrics.py
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from rl_env1 import DynamicPricingEnv
from rl_env_sac import DynamicPricingEnvSAC
from rl_train_dqn import QNetwork as DQNNet
from rl_train_ddqn import QNetwork as DDQNNet
from rl_train_sac import Actor

PRICE_ACTIONS = np.array([200, 300, 400, 500, 600, 700, 800])
GAMMA = 0.99

# -------------------------------------------------
# POLICY RUNNERS (UNCHANGED)
# -------------------------------------------------
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
            mu, _ = actor(torch.FloatTensor(state).unsqueeze(0))
            price = torch.clamp(200 + (mu + 1) * 300, 200, 800).item()

        next_state, reward, done, info = env.step(price)

        log["price"].append(price)
        log["reward"].append(reward)
        log["revenue"].append(info["revenue"])
        log["sold"].append(info["sold"])
        state = next_state

    return log

# -------------------------------------------------
# METRICS
# -------------------------------------------------
def compute_metrics(log):
    prices = np.array(log["price"])
    rewards = np.array(log["reward"])
    revenue = np.array(log["revenue"])
    sold = np.array(log["sold"])

    best_price = prices[np.argmax(prices * sold)] if len(prices) else 0
    regret = (best_price * sold.sum()) - revenue.sum()

    return {
        "total_revenue": revenue.sum(),
        "avg_reward": rewards.mean(),
        "reward_std": rewards.std(),
        "discounted_return": np.sum(rewards * (GAMMA ** np.arange(len(rewards)))),
        "units_sold": sold.sum(),
        "avg_price": prices.mean(),
        "price_volatility": prices.std(),
        "price_change_freq": np.mean(np.diff(prices) != 0),
        "episode_length": len(prices),
        "regret_vs_best_price": max(regret, 0)
    }


def evaluate_all_models(df):
    results = {}

    # BASELINE
    env = DynamicPricingEnv(df, PRICE_ACTIONS)
    baseline_log = run_policy_discrete(env)
    results["Baseline"] = compute_metrics(baseline_log)

    # DQN
    env = DynamicPricingEnv(df, PRICE_ACTIONS)
    dqn = DQNNet(len(env.reset()), len(PRICE_ACTIONS))
    dqn.load_state_dict(torch.load("models/dqn_pricing_model.pt"))
    dqn.eval()
    results["DQN"] = compute_metrics(run_policy_discrete(env, dqn))

    # DDQN
    env = DynamicPricingEnv(df, PRICE_ACTIONS)
    ddqn = DDQNNet(len(env.reset()), len(PRICE_ACTIONS))
    ddqn.load_state_dict(torch.load("models/ddqn_pricing_model.pt"))
    ddqn.eval()
    results["DDQN"] = compute_metrics(run_policy_discrete(env, ddqn))

    # SAC
    env = DynamicPricingEnvSAC(df)
    actor = Actor()
    actor.load_state_dict(torch.load("models/sac_pricing_actor.pt"))
    actor.eval()
    results["SAC"] = compute_metrics(run_policy_sac(env, actor))

    return results
