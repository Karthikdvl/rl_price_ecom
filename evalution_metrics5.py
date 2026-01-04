# =========================================
# RL DYNAMIC PRICING – MODEL EVALUATION
# =========================================

import sys
import os
import pickle
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_env1 import DynamicPricingEnv
from kpi_analysis import compute_kpis
from rl_env_sac import DynamicPricingEnvSAC

# -----------------------------------------
# Load Data
# -----------------------------------------
DATA_PATH = "master_dynamic_pricing_dataset.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])


# -----------------------------------------
# Baseline Evaluation
# -----------------------------------------
def evaluate_baseline(dataframe):
    return compute_kpis(dataframe.copy())


# -----------------------------------------
# RL Model Evaluation
# -----------------------------------------
def evaluate_rl_model(model_path, dataframe, episodes=1):

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    Q = model_data["Q"]
    price_actions = model_data["price_actions"]

    env = DynamicPricingEnv(dataframe.copy(), price_actions)

    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, done, info = env.step(action)

            total_reward += reward
            total_steps += 1

    return compute_kpis(
        env.history.copy(),
        total_reward=total_reward,
        total_steps=total_steps
    )


# -----------------------------------------
# Percentage Change
# -----------------------------------------
def pct_change(val, base):
    if base == 0:
        return 0.0
    return ((val - base) / base) * 100


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":

    baseline = evaluate_baseline(df)

    dqn = evaluate_rl_model("models/dqn_model.pkl", df)
    ddqn = evaluate_rl_model("models/ddqn_model.pkl", df)
    sac = evaluate_rl_model("models/sac_model.pkl", df)

    models = {
        "Baseline": baseline,
        "DQN": dqn,
        "DDQN": ddqn,
        "SAC": sac
    }

    table = []

    for name, m in models.items():
        if name == "Baseline":
            table.append([
                name,
                f"{m['total_revenue']:.2f}",
                "0.00%",
                "0.00%",
                "0.00%",
                "0.00%",
                "0.00%"
            ])
        else:
            table.append([
                name,
                f"{m['total_revenue']:.2f}",
                f"{pct_change(m['total_revenue'], baseline['total_revenue']):+.2f}%",
                f"{pct_change(m['avg_reward'], baseline['avg_reward']):+.2f}%",
                f"{pct_change(m['units_sold'], baseline['units_sold']):+.2f}%",
                f"{pct_change(m['gross_margin_pct'], baseline['gross_margin_pct']):+.2f}%",
                f"{pct_change(m['price_volatility'], baseline['price_volatility']):+.2f}%"
            ])

    headers = [
        "Model",
        "Total Revenue",
        "Revenue Change (%)",
        "Avg Reward Change (%)",
        "Units Sold Change (%)",
        "Margin % Change (%)",
        "Price Volatility Change (%)"
    ]

    print("\n" + "=" * 110)
    print("RL DYNAMIC PRICING – PERFORMANCE EVALUATION (PERCENTAGE)")
    print("=" * 110)
    print(tabulate(table, headers=headers, tablefmt="grid"))

    print("\nNotes:")
    print(" + Positive % → Improvement over baseline")
    print(" - Negative % → Underperformance")
    print(" ↓ Lower price volatility is preferred")
    print("=" * 110)
