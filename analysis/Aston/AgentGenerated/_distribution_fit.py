"""Test: is the issue that 1-min returns are NOT lognormal?
N(d2) assumes lognormal. If empirical returns are fat-tailed (kurtosis > 3),
then the probability of finishing far from spot is HIGHER than N(d2) says,
which makes ATM low-prob bets too rich and high-prob too cheap (S-curve).
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load
from utility import SECONDS_PER_YEAR, ANN_MIN
from scipy import stats

d = load()
spot = d["spot"].sort_values("ts").reset_index(drop=True)
print(f"spot ticks: {len(spot):,}")

# 1-min log returns
s = spot.set_index("ts")
m1 = s["price"].resample("1min").last().dropna()
r1 = np.log(m1 / m1.shift(1)).dropna()
print(f"1-min returns: n={len(r1):,}")

# 15-min log returns
m15 = s["price"].resample("15min").last().dropna()
r15 = np.log(m15 / m15.shift(1)).dropna()

print("\n--- 1-min log returns ---")
print(f"  mean:     {r1.mean()*1e4:+.2f} bps")
print(f"  std:      {r1.std()*1e4:.2f} bps")
print(f"  skew:     {stats.skew(r1):+.3f}")
print(f"  kurtosis: {stats.kurtosis(r1):+.3f}  (Gaussian=0, lognormal-of-ret should be small)")
print(f"  ann σ:    {r1.std() * np.sqrt(ANN_MIN) * 100:.1f}%")

print("\n--- 15-min log returns ---")
print(f"  n:        {len(r15):,}")
print(f"  mean:     {r15.mean()*1e4:+.2f} bps")
print(f"  std:      {r15.std()*1e4:.2f} bps")
print(f"  skew:     {stats.skew(r15):+.3f}")
print(f"  kurtosis: {stats.kurtosis(r15):+.3f}")
print(f"  ann σ:    {r15.std() * np.sqrt(365.25 * 24 * 4) * 100:.1f}%")

print("\n--- 15-min vs Gaussian — tail probabilities ---")
sigma_15 = r15.std()
for k in [1, 1.5, 2, 2.5, 3]:
    actual_tail = ((r15.abs() > k * sigma_15).mean())
    norm_tail = 2 * (1 - stats.norm.cdf(k))
    print(f"  |r| > {k}σ:  empirical={actual_tail:.4f}  Gaussian={norm_tail:.4f}  "
          f"ratio={actual_tail/norm_tail:.2f}")

# Now: per-ticker, compare predicted vs empirical "finish far from spot"
print("\n--- Per-ticker actual outcome by |moneyness at quote time| ---")
print("Theo computes N(d2) assuming Gaussian. If z = log(S/K)/(σ√T) = -1.5,")
print("N(d2) says outcome=YES with prob N(-1.5) ≈ 6.7%.")
print("Empirical?")
from utility import implied_sigma, theo_vec
ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
setts = pd.read_pickle(ROOT / "settlements.pkl")
theo = d["theo"].sort_values("ts")
# Pick T-5m snapshot per ticker, compute z using HAR sigma
near = theo[theo["seconds_to_expiry"].between(295, 305)]
near = near.sort_values(["ticker", "seconds_to_expiry"]).drop_duplicates("ticker", keep="first").copy()
near = near.merge(setts[["ticker", "outcome"]], on="ticker", how="inner")
T = near["seconds_to_expiry"] / SECONDS_PER_YEAR
near["z"] = np.log(near["spot"] / near["strike"]) / (near["sigma"] * np.sqrt(T))
near["theo_recompute"] = theo_vec(near["spot"], near["strike"], near["sigma"], near["seconds_to_expiry"])
near["z_bin"] = pd.cut(near["z"], bins=np.arange(-3, 3.5, 0.5))
g = near.groupby("z_bin", observed=True).agg(
    n=("outcome", "count"),
    actual=("outcome", "mean"),
    theo_pred=("theo_recompute", "mean"),
).reset_index()
g["normal_pred"] = stats.norm.cdf(np.array([float(str(z).split(",")[0][1:]) for z in g["z_bin"]]) + 0.25)
g["ratio"] = g["actual"] / g["theo_pred"].replace(0, np.nan)
print(g.to_string(index=False))

# 5min realized returns for ETH
print("\n--- 5-min log returns ---")
m5 = s["price"].resample("5min").last().dropna()
r5 = np.log(m5 / m5.shift(1)).dropna()
print(f"  n:        {len(r5):,}")
print(f"  mean:     {r5.mean()*1e4:+.2f} bps")
print(f"  std:      {r5.std()*1e4:.2f} bps")
print(f"  skew:     {stats.skew(r5):+.3f}")
print(f"  kurtosis: {stats.kurtosis(r5):+.3f}")

# Q-Q against Gaussian: is the body Gaussian and the tails fat?
print("\n--- 15-min log returns: extreme quantiles ---")
print(f"{'q':>6} {'empirical':>12} {'Gaussian':>12}")
for q in [0.001, 0.005, 0.01, 0.025, 0.05, 0.95, 0.975, 0.99, 0.995, 0.999]:
    print(f"  {q:.3f}  {r15.quantile(q)*1e4:>+10.1f} bps  {stats.norm.ppf(q)*sigma_15*1e4:>+10.1f} bps")
