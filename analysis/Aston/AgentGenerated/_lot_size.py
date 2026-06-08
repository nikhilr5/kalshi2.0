"""Lot size analysis: do larger lots fill at worse prices?
Is there latent capacity at 5-lot, 10-lot, etc?"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
d = load()
book = d["book"]

# Use mid_at_fill for context; bid_size/ask_size for top-of-book depth
print("=" * 78)
print("BOOK DEPTH AT FILL")
print("=" * 78)
print(f"  Top-bid size median: {f['bid_size'].median():.0f}  mean: {f['bid_size'].mean():.0f}  q90: {f['bid_size'].quantile(0.9):.0f}")
print(f"  Top-ask size median: {f['ask_size'].median():.0f}  mean: {f['ask_size'].mean():.0f}  q90: {f['ask_size'].quantile(0.9):.0f}")

# By minutes to close
print("\n--- BOOK DEPTH BY MINUTES TO CLOSE ---")
print(f"{'mins':>8} {'n':>5} {'bid_med':>9} {'ask_med':>9} {'bid_q90':>9} {'ask_q90':>9}")
for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 16)]:
    s = f[(f["mins_to_close"] >= lo) & (f["mins_to_close"] < hi)]
    print(f"  [{lo:>2},{hi:>2})  {len(s):>5}  {s['bid_size'].median():>8.0f}  "
          f"{s['ask_size'].median():>8.0f}  {s['bid_size'].quantile(0.9):>8.0f}  "
          f"{s['ask_size'].quantile(0.9):>8.0f}")

print("\n" + "=" * 78)
print("WHAT ARE 'BIG' KALSHI BOOKS? Look at the recorded book table.")
print("=" * 78)
book["mid"] = (book["yes_bid"] + book["yes_ask"]) / 2
# Show typical book sizes
print(f"  All book snapshots: {len(book):,}")
print(f"  Bid size — median: {book['bid_size'].median():.0f}  mean: {book['bid_size'].mean():.0f}  "
      f"q90: {book['bid_size'].quantile(0.9):.0f}  q99: {book['bid_size'].quantile(0.99):.0f}")
print(f"  Ask size — median: {book['ask_size'].median():.0f}  mean: {book['ask_size'].mean():.0f}  "
      f"q90: {book['ask_size'].quantile(0.9):.0f}  q99: {book['ask_size'].quantile(0.99):.0f}")

print(f"\n  Books with bid_size >= 100: {(book['bid_size'] >= 100).sum():,}  "
      f"({100*(book['bid_size']>=100).mean():.1f}%)")
print(f"  Books with ask_size >= 100: {(book['ask_size'] >= 100).sum():,}  "
      f"({100*(book['ask_size']>=100).mean():.1f}%)")

# How would 5-lot orders fare? Most book snapshots have plenty of room at top
# Implied capacity: most fills are 1-lot vs hundreds of contracts at top.
# Real question is how big a single fill can we have without spread shifting against us.
print("\n--- ESTIMATED MAX 1-FILL SIZE BEFORE PRICE IMPACT ---")
# How many fills do we get back-to-back on the same ticker in <1s?
f_s = f.sort_values(["ticker", "ts"]).copy()
f_s["dt"] = f_s.groupby("ticker")["ts"].diff().dt.total_seconds()
print(f"  Median inter-fill time: {f_s['dt'].median():.1f}s")
print(f"  Q10 inter-fill:         {f_s['dt'].quantile(0.1):.1f}s")
print(f"  Fills <1s apart:        {(f_s['dt']<1).sum():,}  ({100*(f_s['dt']<1).mean():.1f}%)")
print(f"  Fills <5s apart:        {(f_s['dt']<5).sum():,}")

# What's the total fill burst size in clusters?
# Group consecutive fills (per ticker, action) within 1s
f_s["new_cluster"] = (
    (f_s.groupby(["ticker", "action"])["ts"].diff().dt.total_seconds() > 5)
    | (f_s.groupby(["ticker", "action"])["ts"].diff().isna())
).astype(int).cumsum()
clusters = f_s.groupby(["ticker", "action", "new_cluster"]).agg(
    cluster_size=("count", "sum"),
    n=("count", "count"),
    dur=("ts", lambda x: (x.max() - x.min()).total_seconds()),
).reset_index()
print(f"\n  Fill clusters (same ticker+side, <5s spacing): n={len(clusters):,}")
print(f"  Cluster size dist: med={clusters['cluster_size'].median():.0f}  "
      f"q90={clusters['cluster_size'].quantile(0.9):.0f}  "
      f"q99={clusters['cluster_size'].quantile(0.99):.0f}  "
      f"max={clusters['cluster_size'].max():.0f}")
print(f"  Multi-fill clusters (n>=3): {(clusters['n']>=3).sum():,}")
print(f"  Mega clusters  (n>=8):       {(clusters['n']>=8).sum():,}")
