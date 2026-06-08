"""For each fill, what was the age of the resting order at fill time?
And: at fill time, where was theo vs the resting price?"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
d = load()
events = d["events"]
events["ts"] = pd.to_datetime(events["ts"], utc=True, format="ISO8601")

# Match fills to their placed order's last placed_event before fill
# event_type='placed' has order_id. For each fill, find the most recent placed
# event with matching order_id and ticker before fill ts.
# fills.client_order_id matches event_type='placed' that has same client_order_id.
# But fills don't carry order_id directly. Use action+ticker+price match.

# Easier path: order_events has "placed" with price. For each (ticker, action),
# find the most recent placed event prior to each fill where price matches.

placed = events[events["event_type"] == "placed"].copy()
placed = placed[["ts", "order_id", "ticker", "action", "price"]].sort_values("ts")
fills = f.sort_values("ts").copy()
fills["fid"] = np.arange(len(fills))

# merge_asof to find the most recent place (any) per ticker+action+price
# Use a synthetic key
fills["match_key"] = fills["ticker"] + "_" + fills["action"] + "_" + (fills["price"]*1000).round().astype(int).astype(str)
placed["match_key"] = placed["ticker"] + "_" + placed["action"] + "_" + (placed["price"]*1000).round().astype(int).astype(str)

merged = pd.merge_asof(
    fills.sort_values("ts"),
    placed[["ts", "match_key", "order_id"]].rename(columns={"ts": "placed_ts"}).sort_values("placed_ts"),
    by="match_key", left_on="ts", right_on="placed_ts", direction="backward",
    tolerance=pd.Timedelta("15min"),
)
merged["quote_age_s"] = (merged["ts"] - merged["placed_ts"]).dt.total_seconds()
print(f"matched: {merged['quote_age_s'].notna().sum():,}/{len(fills):,}")

print("\n--- QUOTE AGE AT FILL (seconds) ---")
age = merged["quote_age_s"].dropna()
print(f"  Mean:    {age.mean():.1f}s")
print(f"  Median:  {age.median():.1f}s")
print(f"  Q90:     {age.quantile(0.9):.1f}s")
print(f"  Q99:     {age.quantile(0.99):.1f}s")
print(f"  >10s:    {(age>10).mean():.1%}")
print(f"  >30s:    {(age>30).mean():.1%}")
print(f"  >60s:    {(age>60).mean():.1%}")

# P&L by quote age bucket
merged["age_bucket"] = pd.cut(merged["quote_age_s"],
                              bins=[0, 0.5, 2, 5, 10, 30, 60, 300, 1000],
                              include_lowest=True)
g = merged.groupby("age_bucket", observed=True).agg(
    n=("pnl_settle_c", "count"),
    pnl=("pnl_settle_c", lambda x: x.sum()/100),
    per_fill=("pnl_settle_c", "mean"),
    edge=("edge_c", "mean"),
    buy_pct=("action", lambda x: (x=="buy").mean()),
).reset_index()
print("\n--- P&L BY QUOTE AGE ---")
print(g.to_string(index=False))

# Per action
print("\n--- BY ACTION ---")
for action in ["buy", "sell"]:
    print(f"\n  {action}:")
    g = merged[merged["action"]==action].groupby("age_bucket", observed=True).agg(
        n=("pnl_settle_c", "count"),
        pnl=("pnl_settle_c", lambda x: x.sum()/100),
        per_fill=("pnl_settle_c", "mean"),
        edge=("edge_c", "mean"),
    ).reset_index()
    print(g.to_string(index=False))
