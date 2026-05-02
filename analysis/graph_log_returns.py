from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utility import AnalysisUtils


if __name__ == "__main__":
    u = AnalysisUtils()

    df = u.load_one_market_snapshot(datetime(2026, 4, 24, 0, 0), datetime(2026, 4, 30, 16, 0), event_prefix="KXBTCD")

    # spot_mid is the same across all strikes — just need one series per timestamp
    spot = df.drop_duplicates(subset=["ts"])[["ts", "spot_mid"]].copy()
    spot = spot.sort_values("ts").reset_index(drop=True)

    # Snap to 5-min grid
    grid = spot.set_index("ts").resample("5min").last().dropna().reset_index()

    # Log returns in bps
    grid["logret_bps"] = 10_000 * np.log(grid["spot_mid"] / grid["spot_mid"].shift(1))

    # Histogram rounded to nearest whole bps
    s = grid["logret_bps"].replace([np.inf, -np.inf], np.nan).dropna()
    counts = s.round().astype(int).value_counts().sort_index()

    # Fit normal distribution
    mu, sigma = stats.norm.fit(s)

    plt.figure(figsize=(11, 5))
    plt.bar(counts.index, counts.values, width=0.9, edgecolor="black", linewidth=0.3, label="Observed")

    # Overlay fitted normal curve scaled to histogram
    x = np.linspace(s.min(), s.max(), 200)
    pdf = stats.norm.pdf(x, mu, sigma) * len(s)  # scale to match frequency counts
    plt.plot(x, pdf, "r-", linewidth=2, label=f"Normal fit (μ={mu:.1f}, σ={sigma:.1f})")

    plt.xlabel("Log return (bps, rounded)")
    plt.ylabel("Frequency")
    plt.title("Histogram of spot_mid log returns (bps) — 5-min grid")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
