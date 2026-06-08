"""Counterfactual: what would P&L be with different edge values?
For each fill, ask: would the strategy have posted at this fill price
with edge X? If our edge is bigger than current (X > observed_edge),
we wouldn't have posted there — counterfactual = drop the fill.
If smaller (X < observed_edge), we WOULD have posted at theo-X — but
would the counterparty have hit us? Only if the same order from them
would have hit at theo-X. So we can model:
  - drop_fills with edge_c < X (because we wouldn't have been there)
  - approximate that fills with edge_c >= X would have happened at price = theo - X
    (because someone willing to buy at the worse price would also buy at the better price)

This is conservative on the upside (we might have gotten MORE fills with tighter edges)
and we keep current bid/sell mix.
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]
f = f.dropna(subset=["pnl_settle_c", "edge_c", "theo", "outcome"])
n_days = f["date"].nunique()

def pnl_with_edge(f, edge_bid_c, edge_ask_c):
    """Each fill: if edge_c >= our_edge for its side, we'd have posted at
    price = theo +/- our_edge. Recalc P&L at that price.
    Drop fills where edge_c < our_edge (we wouldn't have been there)."""
    out = []
    for action, edge in [("buy", edge_bid_c), ("sell", edge_ask_c)]:
        s = f[f["action"] == action].copy()
        # Drop fills with edge_c < our edge (we'd have been outside)
        s = s[s["edge_c"] >= edge]
        # New fill price = theo - edge (buy) or theo + edge (sell)
        if action == "buy":
            new_price = s["theo"] - edge/100
            new_pnl = (s["outcome"] - new_price) * 100 * s["count"]
        else:
            new_price = s["theo"] + edge/100
            new_pnl = (new_price - s["outcome"]) * 100 * s["count"]
        out.append(dict(action=action, n=len(s),
                        pnl=new_pnl.sum()/100,
                        pnl_per_fill=new_pnl.mean(),
                        per_contract=new_pnl.sum()/s["count"].sum() if s["count"].sum() > 0 else 0))
    return pd.DataFrame(out)

print("=" * 78)
print("EDGE GRID SEARCH — held-to-close, conditional on existing fills")
print("=" * 78)
# Coarse grid
results = []
for eb in [3, 5, 6, 7, 8, 9, 10, 12]:
    for ea in [3, 4, 5, 6, 7, 8, 10]:
        r = pnl_with_edge(f, eb, ea)
        total = r["pnl"].sum()
        n = r["n"].sum()
        results.append(dict(edge_bid=eb, edge_ask=ea, total_pnl=total,
                            per_day=total/n_days, n=int(n)))
R = pd.DataFrame(results)
print("\nTotal P&L (held-to-close, $) by edge grid (rows=bid, cols=ask):")
piv = R.pivot(index="edge_bid", columns="edge_ask", values="per_day").round(2)
print(piv.to_string())

print("\nFills retained (n):")
piv = R.pivot(index="edge_bid", columns="edge_ask", values="n")
print(piv.to_string())

# Find top configs by per-day P&L
top = R.sort_values("per_day", ascending=False).head(10)
print("\n--- TOP 10 EDGE CONFIGS ---")
print(top.to_string(index=False))

# Sweet-spot check: also report Sharpe-ish (mean / sd of daily P&L)
print("\n--- PER-DAY P&L STABILITY at top config ---")
top1 = top.iloc[0]
eb, ea = int(top1["edge_bid"]), int(top1["edge_ask"])
for action, edge in [("buy", eb), ("sell", ea)]:
    s = f[(f["action"] == action) & (f["edge_c"] >= edge)].copy()
    if action == "buy":
        s["new_pnl"] = (s["outcome"] - (s["theo"] - edge/100)) * 100 * s["count"]
    else:
        s["new_pnl"] = ((s["theo"] + edge/100) - s["outcome"]) * 100 * s["count"]
    daily = s.groupby("date")["new_pnl"].sum() / 100
    print(f"  {action}: n={len(s):,}  mean=${daily.mean():+.2f}  std=${daily.std():.2f}  "
          f"days={len(daily)}")
