import pandas as pd
import numpy as np

def compute_full_kpis(df: pd.DataFrame) -> dict:
    kpis = {}

    # ---------------------------
    # BASIC BUSINESS KPIs
    # ---------------------------
    kpis["total_revenue"] = df["line_revenue"].sum()
    kpis["total_units"] = df["quantity"].sum()
    kpis["num_orders"] = df["order_id"].nunique()
    kpis["unique_users"] = df["user_id"].nunique()

    kpis["aov"] = kpis["total_revenue"] / kpis["num_orders"]
    kpis["revenue_per_user"] = kpis["total_revenue"] / kpis["unique_users"]

    # ---------------------------
    # PRICING KPIs
    # ---------------------------
    df["price_realization"] = df["net_price"] / df["list_price"]
    kpis["avg_price_realization"] = df["price_realization"].mean()

    kpis["price_volatility"] = df.groupby("sku_id")["net_price"].std().mean()

    discounted_orders = df[df["net_price"] < df["list_price"]]
    kpis["discount_dependency_ratio"] = len(discounted_orders) / len(df)

    # ---------------------------
    # MARGIN KPIs
    # ---------------------------
    df["total_cost"] = df["base_cost"] * df["quantity"]
    kpis["gross_margin"] = (df["line_revenue"] - df["total_cost"]).sum()
    kpis["gross_margin_pct"] = kpis["gross_margin"] / kpis["total_revenue"]

    # ---------------------------
    # DEMAND & ELASTICITY KPIs
    # ---------------------------
    df = df.sort_values("order_date")
    df["price_pct_change"] = df["net_price"].pct_change()
    df["qty_pct_change"] = df["quantity"].pct_change()

    elasticity = (
        df["qty_pct_change"] / df["price_pct_change"]
    ).replace([np.inf, -np.inf], np.nan)

    kpis["avg_price_elasticity"] = elasticity.mean()

    # ---------------------------
    # INVENTORY KPIs
    # ---------------------------
    kpis["stockout_rate"] = (df["inventory"] == 0).mean()
    kpis["inventory_turnover"] = (
        df["quantity"].sum() / df["inventory"].replace(0, np.nan).mean()
    )

    # ---------------------------
    # CUSTOMER KPIs
    # ---------------------------
    repeat_users = df.groupby("user_id").filter(lambda x: len(x) > 1)
    kpis["repeat_user_ratio"] = repeat_users["user_id"].nunique() / df["user_id"].nunique()

    kpis["return_rate"] = df["daily_returns"].sum() / df["quantity"].sum()

    # ---------------------------
    # COMPETITOR KPIs
    # ---------------------------
    df["price_gap"] = df["net_price"] - df["comp_avg_price"]
    kpis["avg_competitor_price_gap"] = df["price_gap"].mean()

    kpis["competitor_undercut_rate"] = (df["net_price"] > df["comp_avg_price"]).mean()

    # ---------------------------
    # RATING KPIs
    # ---------------------------
    kpis["avg_rating"] = df["avg_rating"].mean()
    kpis["rating_coverage"] = (df["num_reviews"] > 0).mean()

    return kpis
