import numpy as np
import pandas as pd

class DynamicPricingEnvSAC:
    """
    Continuous-action pricing environment for SAC
    Action ∈ [min_price, max_price]
    """

    def __init__(self, df, min_price=200, max_price=800):
        self.df = df.sort_values("order_date")
        self.min_price = min_price
        self.max_price = max_price

        self.daily = (
            self.df.groupby(["sku_id", "order_date_only"])
            .agg(
                demand=("quantity", "sum"),
                avg_price=("net_price", "mean"),
                stock_level=("stock_level", "mean"),
                line_revenue=("line_revenue", "sum"),
            )
            .reset_index()
        )

        self.episode_length = len(self.daily)
        self.reset()

    def reset(self):
        self.idx = 0
        self.inventory = 500
        return self._get_state()

    def _get_state(self):
        row = self.daily.iloc[self.idx]
        demand_norm = row["demand"] / max(self.daily["demand"].max(), 1)
        return np.array(
            [
                row["avg_price"],
                demand_norm,
                self.inventory,
            ],
            dtype=np.float32
        )

    def step(self, price):
        row = self.daily.iloc[self.idx]

        price = float(np.clip(price, self.min_price, self.max_price))

        base_demand = row["demand"]
        price_ratio = price / max(row["avg_price"], 1e-3)
        demand = base_demand * np.exp(-0.5 * (price_ratio - 1))

        sold = min(int(round(max(demand, 0))), self.inventory)
        revenue = sold * price
        self.inventory -= sold

        reward = revenue

        self.idx += 1
        done = self.idx >= self.episode_length or self.inventory <= 0

        next_state = self._get_state() if not done else None

        info = {"price": price, "sold": sold, "revenue": revenue}
        return next_state, reward, done, info
