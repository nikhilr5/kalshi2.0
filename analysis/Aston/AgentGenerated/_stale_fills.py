"""How many fills are stale (theo had moved >tolerance from quote price
before the fill landed)? What's the cost?"""
import sys, pandas as pd, numpy as np, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from utility import bootstrap_ci
from _loader import load

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
mo = pd.read_pickle(ROOT / "markouts.pkl")
f = f.merge(mo[["fid", "markout_60s", "markout_120s"]], left_index=True, right_on="fid")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]

# Stale: theo at fill time is > edge + tolerance away from fill price
# i.e. theo says we'd have repriced if we'd known
# For a buy at price P with theo T: stale if T-P > edge_bid (7c) + tolerance (1c) = 8c
# So if edge_c > 8c on a buy = filled at WORSE price than current theo would post

# But we always *try* to fill at theo-edge. Stale only matters if the THEO at fill is far
# from where we POSTED. We don't track posted theo per fill.
# Best proxy: if edge_c (= theo - price for buy, price - theo for sell) is well above
# the strategy edge, that means theo moved AWAY from us after posting (favorable for us → not stale).
# Stale = edge_c is NEGATIVE (theo moved against us, but the order still filled).

print("=" * 78)
print("STALE FILL ANALYSIS")
print("=" * 78)
# Bucket by edge at fill
print("\n--- BREAKDOWN BY EDGE AT FILL (cents) ---")
buckets = [(-100, -3), (-3, 0), (0, 3), (3, 5), (5, 7), (7, 9), (9, 100)]
print(f"{'edge_range':>14} {'n':>6} {'pct':>5} {'pnl_d':>9} {'per_fill':>9} {'mo_60s':>8} {'buy_pct':>8}")
for lo, hi in buckets:
    s = f[(f["edge_c"] >= lo) & (f["edge_c"] < hi)].dropna(subset=["pnl_settle_c"])
    if len(s) == 0: continue
    pct = 100 * len(s) / len(f.dropna(subset=["edge_c"]))
    p = s["pnl_settle_c"].sum() / 100
    bp = (s["action"] == "buy").mean()
    print(f"  [{lo:>3},{hi:>3})  {len(s):>6}  {pct:>4.1f}%  {p:>+8.2f}  "
          f"{s['pnl_settle_c'].mean():>+8.2f}c  {s['markout_60s'].mean():>+7.2f}c  {bp:>7.1%}")

# Show what the strategy *should* have done
print("\n--- IF WE COULD ELIMINATE NEGATIVE-EDGE FILLS ---")
# 0 = if those fills never happened
neg = f[(f["edge_c"] < 0) & f["pnl_settle_c"].notna()]
print(f"  Negative-edge fills: {len(neg):>5}  pnl=${neg['pnl_settle_c'].sum()/100:+.2f}")
print(f"  Net loss avoidable:  ${-neg['pnl_settle_c'].sum()/100:+.2f}  over 11 days")
print(f"  Per-day avoidable:   ${-neg['pnl_settle_c'].sum()/100/11:+.2f}")

# And by edge < +2c (still suspicious)
sus = f[(f["edge_c"] < 2) & f["pnl_settle_c"].notna()]
print(f"  Edge < +2c fills:    {len(sus):>5}  pnl=${sus['pnl_settle_c'].sum()/100:+.2f}  "
      f"per-day=${sus['pnl_settle_c'].sum()/100/11:+.2f}")

# Now: stale fills by side. Are buys overrepresented?
print("\n--- STALE FILLS BY SIDE ---")
for action in ["buy", "sell"]:
    sub = f[(f["action"] == action) & f["pnl_settle_c"].notna()]
    neg = sub[sub["edge_c"] < 0]
    print(f"  {action}:  total={len(sub):>5}  stale (edge<0)={len(neg):>5}  "
          f"pct={100*len(neg)/len(sub):.1f}%  pnl_lost=${neg['pnl_settle_c'].sum()/100:+.2f}")

# Time-of-quote latency proxy: theo was last updated at theo_ts.
# Stale means theo had moved between theo_ts and fill_ts but we hadn't repriced.
# The merge_asof backward returns the most recent theo BEFORE fill — so it's
# what would have been used to post. Not exactly "stale", but informative.
# Look at fill price vs current theo from book BBO
print("\n--- HOW OFTEN DID WE BUY AT PRICE > BBO MID (took the ask) ---")
print("Aston posts post-only, so we should only get filled when someone takes US.")
print("If our price > current bid (for buys), that means market dropped after we posted.")
f["filled_above_bbo"] = (
    ((f["action"] == "buy") & (f["price"] > f["book_bid"]))
    | ((f["action"] == "sell") & (f["price"] < f["book_ask"]))
)
print(f"  Fills above-the-book (favorable side):  {f['filled_above_bbo'].sum():,}  "
      f"({100*f['filled_above_bbo'].mean():.1f}%)")

# Most useful: distribution of fill price RELATIVE TO instantaneous mid
f["price_vs_mid"] = np.where(
    f["action"] == "buy",
    (f["price"] - f["mid_at_fill"]) * 100,  # +ve = bought above mid (bad)
    (f["mid_at_fill"] - f["price"]) * 100,  # +ve = sold below mid (bad)
)
print("\n--- FILL PRICE vs INSTANT MID (cents, +ve = filled at WORSE than mid) ---")
print(f"  Mean:     {f['price_vs_mid'].mean():+.2f}c")
print(f"  Median:   {f['price_vs_mid'].median():+.2f}c")
print(f"  Q90:      {f['price_vs_mid'].quantile(0.9):+.2f}c  (top 10% bleed)")
for action in ["buy", "sell"]:
    s = f[f["action"] == action]
    print(f"  {action:>4}:  mean={s['price_vs_mid'].mean():+.2f}c  median={s['price_vs_mid'].median():+.2f}c")

# Were the bad fills also in the late market?
print("\n--- BAD FILLS BY MINUTES TO CLOSE ---")
neg = f[(f["edge_c"] < 0) & f["pnl_settle_c"].notna()]
print(neg["mins_to_close"].describe())
