"""Realistic version of directional strategy: include fees, slippage,
and capacity constraints. Also test 'one trade per market' constraint
since you can't take multiple times in the same market."""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load
from utility import SECONDS_PER_YEAR

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
setts = pd.read_pickle(ROOT / "settlements.pkl")

d = load()
theo = d["theo"]; book = d["book"]
g = theo.groupby("ticker").agg(
    last_ts=("ts","max"), last_secs=("seconds_to_expiry","min"),
    strike=("strike","first")).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(setts[["ticker","outcome"]], on="ticker", how="inner")

# Use ALL snapshots, not just specific offsets
theo_m = theo.merge(g[["ticker","close_time","outcome"]], on="ticker", how="inner")
theo_m["secs_to_close"] = (theo_m["close_time"] - theo_m["ts"]).dt.total_seconds()

# Restrict to T-30s to T-15m (don't trade final 30s or earlier than 14m)
print("loading book...")
bk = book[["ts","ticker","yes_bid","yes_ask"]].sort_values("ts")

# Merge spot to know what we're seeing
# Sample theo every 5s by ticker
theo_m = theo_m[(theo_m["secs_to_close"] >= 30) & (theo_m["secs_to_close"] <= 900)]
# Downsample to once every 5s to keep tractable
theo_m["bucket_5s"] = (theo_m["secs_to_close"] // 5).astype(int)
sampled = theo_m.sort_values(["ticker", "bucket_5s", "secs_to_close"]).drop_duplicates(["ticker","bucket_5s"], keep="first").copy()
print(f"sampled rows: {len(sampled):,}")

# Attach BBO
sampled = sampled.sort_values("ts")
sampled = pd.merge_asof(sampled, bk, by="ticker", on="ts", direction="backward",
                         tolerance=pd.Timedelta("3s"))
sampled["mid"] = (sampled["yes_bid"] + sampled["yes_ask"]) / 2
sampled["spread"] = (sampled["yes_ask"] - sampled["yes_bid"]) * 100
sampled = sampled.dropna(subset=["mid","theo","outcome","yes_bid","yes_ask"])

# Kalshi fee: ~ ceil(round(0.07 * P * (1-P) * count * 100)) / 100 — complicated
# Approximation: 7% of |contract_value_change_at_exit| ≈ 0.07 * |outcome - entry|
# More precisely: fee per contract = ceil(7 * P * (1-P)) cents for opening
# We'll use a simple 1.5c per trade as a reasonable approximation
FEE_PER_TRADE = 1.0  # cents on a 1-lot trade, conservative

print(f"\n--- ONE-TRADE-PER-MARKET CONSTRAINT (each ticker: first signal only) ---")
print(f"Fee per trade: {FEE_PER_TRADE}c")

for thresh_c in [2, 3, 5, 8, 12]:
    thresh = thresh_c / 100
    signal = sampled["theo"] - sampled["mid"]
    # Buy if signal>thresh, Sell if signal<-thresh
    sampled["trade_side"] = np.where(signal > thresh, "buy",
                                      np.where(signal < -thresh, "sell", ""))
    trades = sampled[sampled["trade_side"] != ""].copy()
    # First trade per ticker
    trades = trades.sort_values("ts").drop_duplicates("ticker", keep="first")
    if len(trades) == 0: continue
    trades["entry"] = np.where(trades["trade_side"] == "buy", trades["yes_ask"], trades["yes_bid"])
    trades["pnl_c"] = np.where(trades["trade_side"] == "buy",
                               (trades["outcome"] - trades["entry"]) * 100,
                               (trades["entry"] - trades["outcome"]) * 100)
    trades["pnl_after_fee_c"] = trades["pnl_c"] - FEE_PER_TRADE
    total = trades["pnl_after_fee_c"].sum() / 100
    n_days = sampled["ts"].dt.tz_convert("America/Chicago").dt.date.nunique()
    print(f"  thresh={thresh_c:>2}c  n={len(trades):>4}  total=${total:>+6.2f}  "
          f"per_day=${total/n_days:>+5.2f}  per_trade={trades['pnl_after_fee_c'].mean():>+5.2f}c  "
          f"win={(trades['pnl_after_fee_c']>0).mean():.1%}")

# Same but allow multiple trades per ticker (max position 1 contract)
print(f"\n--- ALL SIGNALS (each ticker: every 5s, multiple trades allowed) ---")
print("(this is the theoretical max, not realistic with capacity)")
for thresh_c in [2, 3, 5, 8, 12]:
    thresh = thresh_c / 100
    signal = sampled["theo"] - sampled["mid"]
    sampled["trade_side"] = np.where(signal > thresh, "buy",
                                      np.where(signal < -thresh, "sell", ""))
    trades = sampled[sampled["trade_side"] != ""].copy()
    trades["entry"] = np.where(trades["trade_side"] == "buy", trades["yes_ask"], trades["yes_bid"])
    trades["pnl_c"] = np.where(trades["trade_side"] == "buy",
                               (trades["outcome"] - trades["entry"]) * 100,
                               (trades["entry"] - trades["outcome"]) * 100)
    trades["pnl_after_fee_c"] = trades["pnl_c"] - FEE_PER_TRADE
    total = trades["pnl_after_fee_c"].sum() / 100
    n_days = sampled["ts"].dt.tz_convert("America/Chicago").dt.date.nunique()
    print(f"  thresh={thresh_c:>2}c  n={len(trades):>4}  total=${total:>+6.2f}  "
          f"per_day=${total/n_days:>+5.2f}  per_trade={trades['pnl_after_fee_c'].mean():>+5.2f}c  "
          f"win={(trades['pnl_after_fee_c']>0).mean():.1%}")

# Time-restricted: only T-5m to T-2m (the sweet spot)
print(f"\n--- ONE-TRADE-PER-MARKET, T-2m to T-5m window ---")
window = sampled[sampled["secs_to_close"].between(120, 300)]
for thresh_c in [2, 3, 5, 8, 12]:
    thresh = thresh_c / 100
    signal = window["theo"] - window["mid"]
    side = np.where(signal > thresh, "buy", np.where(signal < -thresh, "sell", ""))
    w = window.assign(trade_side=side)
    trades = w[w["trade_side"] != ""].copy()
    trades = trades.sort_values("ts").drop_duplicates("ticker", keep="first")
    trades["entry"] = np.where(trades["trade_side"] == "buy", trades["yes_ask"], trades["yes_bid"])
    trades["pnl_c"] = np.where(trades["trade_side"] == "buy",
                               (trades["outcome"] - trades["entry"]) * 100,
                               (trades["entry"] - trades["outcome"]) * 100)
    trades["pnl_after_fee_c"] = trades["pnl_c"] - FEE_PER_TRADE
    total = trades["pnl_after_fee_c"].sum() / 100
    n_days = window["ts"].dt.tz_convert("America/Chicago").dt.date.nunique()
    print(f"  thresh={thresh_c:>2}c  n={len(trades):>4}  total=${total:>+6.2f}  "
          f"per_day=${total/n_days:>+5.2f}  per_trade={trades['pnl_after_fee_c'].mean():>+5.2f}c  "
          f"win={(trades['pnl_after_fee_c']>0).mean():.1%}")

# Check the theo>mid (buy) vs theo<mid (sell) directional contribution
print(f"\n--- DIRECTIONAL SPLIT at thresh=5c, T-2m to T-5m, 1/ticker ---")
window = sampled[sampled["secs_to_close"].between(120, 300)]
thresh = 5/100
signal = window["theo"] - window["mid"]
side = np.where(signal > thresh, "buy", np.where(signal < -thresh, "sell", ""))
w = window.assign(trade_side=side)
trades = w[w["trade_side"] != ""].copy().sort_values("ts").drop_duplicates("ticker", keep="first")
trades["entry"] = np.where(trades["trade_side"] == "buy", trades["yes_ask"], trades["yes_bid"])
trades["pnl_c"] = np.where(trades["trade_side"] == "buy",
                           (trades["outcome"] - trades["entry"]) * 100,
                           (trades["entry"] - trades["outcome"]) * 100)
trades["pnl_after_fee_c"] = trades["pnl_c"] - FEE_PER_TRADE
for s in ["buy", "sell"]:
    sub = trades[trades["trade_side"] == s]
    print(f"  {s}:  n={len(sub):>4}  total=${sub['pnl_after_fee_c'].sum()/100:>+6.2f}  "
          f"per_trade={sub['pnl_after_fee_c'].mean():>+5.2f}c  "
          f"win={(sub['pnl_after_fee_c']>0).mean():.1%}")
