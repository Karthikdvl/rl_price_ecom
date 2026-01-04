# rl_env.py
import numpy as np
import pandas as pd


class DynamicPricingEnv:
    """
    Custom environment for dynamic pricing (no Gym dependency)
    """

    def __init__(self, df: pd.DataFrame, price_actions):
        self.df = df.sort_values("order_date")
        self.price_actions = np.array(price_actions)
        self.n_actions = len(price_actions)

        # Daily SKU-level aggregation
        self.daily = (
            self.df.groupby(["sku_id", "order_date_only"])
            .agg(
                demand=("quantity", "sum"),
                avg_price=("net_price", "mean"),
                stock_level=("stock_level", "mean"),
                comp_avg_price=("comp_avg_price", "mean"),
                daily_sku_views=("daily_sku_views", "sum"),
                line_revenue=("line_revenue", "sum"),
            )
            .reset_index()
        )

        self.episode_length = len(self.daily)
        self.reset()

    def reset(self):
        self.idx = 0
        self.inventory = 500  # fixed initial inventory
        return self._get_state()

    def _get_state(self):
        row = self.daily.iloc[self.idx]

        price_gap = (
            row["avg_price"] - row["comp_avg_price"]
            if not np.isnan(row["comp_avg_price"])
            else 0.0
        )

        demand_norm = row["demand"] / max(self.daily["demand"].max(), 1)

        return np.array(
            [
                row["avg_price"],
                price_gap,
                row["stock_level"],
                demand_norm,
                self.inventory,
            ],
            dtype=np.float32,
        )

    def step(self, action_idx):
        row = self.daily.iloc[self.idx]
        chosen_price = self.price_actions[action_idx]

        # Demand response model
        base_demand = row["demand"]
        price_ratio = chosen_price / max(row["avg_price"], 1e-3)
        demand = base_demand * np.exp(-0.5 * (price_ratio - 1))

        demand = max(0, int(round(demand)))
        sold = min(demand, self.inventory)

        revenue = sold * chosen_price
        self.inventory -= sold

        # # Reward: revenue − holding cost
        # reward = revenue - 0.1 * self.inventory
        
        # Revenue-based reward (pure)
        reward = revenue


        self.idx += 1
        done = self.idx >= self.episode_length or self.inventory <= 0

        next_state = self._get_state() if not done else None

        info = {
            "revenue": revenue,
            "sold": sold,
            "demand": demand,
            "price": chosen_price,
        }

        return next_state, reward, done, info
