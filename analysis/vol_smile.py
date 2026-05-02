from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import os 
from utility import AnalysisUtils


u = AnalysisUtils()
reload = False

if not os.path.exists('./data/vol_smile.csv') or reload:
    df = u.load_market_snapshots(datetime(2026, 4, 26, 0, 0), datetime(2026, 4, 30, 16, 0), event_prefix="KXBTCD-26MAY0117")
    df = df[df['expiry_type'] == 'weekly']

    df["T"] = (pd.to_datetime(df["close_time"], utc=True) - df["ts"]).dt.total_seconds() / (365.25 * 24 * 3600)


    df["bid_iv"] = df.apply(lambda r: u.implied_vol_binary(r["kalshi_yes_bid"], r["spot_bid"], r["strike"], r["T"], 0.043), axis=1)
    df["ask_iv"] = df.apply(lambda r: u.implied_vol_binary(r["kalshi_yes_ask"], r["spot_ask"], r["strike"], r["T"], 0.043), axis=1)
    df["mid_iv"] = (df["bid_iv"] + df["ask_iv"]) / 2
    df['otm_pct'] = df['strike'] / df['spot_mid'] - 1
    df = df[["T", "ticker", "otm_pct", "strike", "ts", "kalshi_yes_bid", "kalshi_yes_ask", "spot_bid", "spot_ask", "spot_mid", "bid_iv", "ask_iv", "mid_iv"]]

    # For each timestamp, fit smile on mid IV
    for ts, full_group in df.groupby("ts"):
        nearby = full_group[full_group['otm_pct'].abs() < 0.04]
        if nearby.empty:
            continue

        # fit on nearby strikes using mid IV
        (a, b, c) = u.fit_vol_smile(nearby["strike"].values, nearby["mid_iv"].values)

        # evaluate fitted IV for ALL strikes at this ts
        all_strikes = full_group["strike"].values
        all_fitted = a * all_strikes**2 + b * all_strikes + c
        df.loc[full_group.index, "fitted_mid_iv"] = all_fitted

    # smooth per strike for multiple spans
    SPANS = [10, 20, 30, 60, 100, 120, 150]
    r = 0.04
    S = df["spot_mid"].values
    K = df["strike"].values
    T = df["T"].values

    for span in SPANS:
        col_iv = f"smoothed_mid_iv_{span}"
        col_theo = f"theo_fitted_{span}"
        df[col_iv] = df.groupby("strike")["fitted_mid_iv"].transform(lambda x: x.ewm(span=span).mean())
        sigma = df[col_iv].values
        d2 = (np.log(S / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        df[col_theo] = stats.norm.cdf(d2)

    # default aliases for backward compat
    df["smoothed_mid_iv"] = df["smoothed_mid_iv_60"]
    df["theo_fitted"] = df["theo_fitted_60"]


    print(df.columns)
    df.to_csv('./data/vol_smile.csv')
else:
    df = pd.read_csv('./data/vol_smile.csv')




## Graph: Average smoothed IV per strike + fitted smile
span = 10
col_iv = f"smoothed_mid_iv_{span}"
if col_iv not in df.columns:
    col_iv = "smoothed_mid_iv"

# Compute mean smoothed IV per strike
avg_iv = df.groupby("strike")[col_iv].mean().dropna()
avg_iv = avg_iv[avg_iv > 0]

strikes_np = avg_iv.index.values
ivs_np = avg_iv.values

# Fit smile on the averages
mask = ivs_np > 0
(a, b, c) = np.polyfit(strikes_np[mask], ivs_np[mask], 2)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(strikes_np, ivs_np * 100, s=30, color="#facc15", zorder=3, label="Avg Smoothed IV")

k_range = np.linspace(strikes_np.min(), strikes_np.max(), 200)
fitted = (a * k_range**2 + b * k_range + c) * 100
ax.plot(k_range, fitted, color="#8b5cf6", linewidth=2, label="Fitted smile")

ax.set_xlabel("Strike")
ax.set_ylabel("IV (%)")
ax.set_title(f"Average Smoothed IV per Strike (span={span})")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
plt.show()

# u.graph_strike(df, 75500.0, iv_cols=["smoothed_mid_iv"])