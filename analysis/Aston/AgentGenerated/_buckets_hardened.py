"""Harden the settlement-bucket edge analysis.

settlement_buckets.py shows sell-edge positive / buy-edge negative across
mid-range price buckets. Three things it lacks before that's a sizing tool:

  1. ticker-clustered bootstrap CI per (price-bucket, side) — fills are not
     independent; the market is the unit. Which buckets actually clear zero?
  2. time-to-close split — the full study found HAR's edge is a last-2-min /
     deep-OTM effect. Is the sell edge late-window only?
  3. markout-based edge — settlement edge can be "adversely selected into a
     short, but the binary happened to resolve my way." Mid-markout at +30s
     measures whether the *price* moved against us right after the fill, which
     is the cleaner adverse-selection signal.

Maker-only book (gross = net). Rerunnable.
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import list_eligible_dbs, fetch_settlements_from_api
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/Aston")
from kalshi_api import KalshiAPI

SERIES, CUTOFF = "KXETH15M", "26MAY15"
PX_BINS = np.arange(0.1, 0.91, 0.1)
N_BOOT = 5000
RNG = np.random.default_rng(42)
SETTLE_CACHE = "/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/.settlements_cache.json"


# ----- load fills + attach +30s markout PER DAY (don't hold all books in RAM;
#       concatenating 23 days of kalshi_book = ~80M rows and swap-thrashes) -----
fill_parts = []
for p in list_eligible_dbs(SERIES, CUTOFF):
    conn = sqlite3.connect(str(p))
    try:
        f = pd.read_sql(
            "SELECT ts,ticker,action,side,count,price,is_taker FROM fills", conn)
        b = pd.read_sql("SELECT ts,ticker,yes_bid,yes_ask FROM kalshi_book", conn)
    except Exception:
        conn.close()
        continue
    conn.close()
    if f.empty:
        continue
    f["ts"] = pd.to_datetime(f["ts"], utc=True, format="ISO8601")
    f = f[f["is_taker"] == 0].copy()
    if f.empty:
        continue
    if not b.empty:
        b["ts"] = pd.to_datetime(b["ts"], utc=True, format="ISO8601")
        b["mid"] = (b["yes_bid"] + b["yes_ask"]) / 2
        b = b.sort_values("ts")
        f = f.sort_values("ts")
        f["mk_ts"] = f["ts"] + pd.Timedelta(seconds=30)
        mk = pd.merge_asof(f[["mk_ts", "ticker"]].sort_values("mk_ts"),
                           b[["ts", "ticker", "mid"]],
                           left_on="mk_ts", right_on="ts", by="ticker",
                           direction="backward", tolerance=pd.Timedelta(seconds=30))
        f["mid_30s"] = mk["mid"].values
    else:
        f["mid_30s"] = np.nan
    fill_parts.append(f.drop(columns=["mk_ts"], errors="ignore"))

fills = pd.concat(fill_parts, ignore_index=True)
fills = fills[fills["ts"] >= "2026-05-15"].copy()

# settlements
s = fetch_settlements_from_api(fills["ticker"].unique().tolist(),
                               KalshiAPI(), cache_path=SETTLE_CACHE)
fills["outcome"] = fills["ticker"].map(s)
fills = fills[fills["outcome"].notna()].copy()

# settlement edge ($/contract), maker so gross == net
fills["edge"] = np.where(fills["action"] == "buy",
                         fills["outcome"] - fills["price"],
                         fills["price"] - fills["outcome"])
# buy gains if mid rises above our price; sell gains if mid falls below it
fills["markout_30s"] = np.where(fills["action"] == "buy",
                                fills["mid_30s"] - fills["price"],
                                fills["price"] - fills["mid_30s"])

# time-to-close: market closes at the :00/:15/:30/:45 after open; 15m markets
# close on the quarter-hour. ttc = seconds from fill to the ticker's close.
# Close time ~ ceil(ts) to next 15-min boundary that matches the ticker hour.
# Simpler + robust: per ticker, close = max fill ts rounded up to 15min.
fills["close"] = fills["ts"].dt.ceil("15min")
fills["ttc_s"] = (fills["close"] - fills["ts"]).dt.total_seconds()

fills = fills[fills["price"].between(0.1, 0.9)].copy()
fills["px_bin"] = pd.cut(fills["price"], PX_BINS)


def cluster_ci(df, col, n=N_BOOT):
    """Ticker-clustered bootstrap mean + 95% CI of `col`."""
    per = df.groupby("ticker")[col].mean().values
    if len(per) < 2:
        return np.nan, np.nan, np.nan, len(per)
    means = np.empty(n)
    for i in range(n):
        means[i] = RNG.choice(per, size=len(per), replace=True).mean()
    return per.mean(), np.quantile(means, .025), np.quantile(means, .975), len(per)


# ===== 1. per-bucket settlement edge with clustered CI =====
print("=" * 84)
print("1.  SETTLEMENT EDGE per (price-bucket, side) — ticker-clustered 95% CI")
print("=" * 84)
print(f"  {'bucket':<12}{'side':<6}{'edge':>9}{'lo':>9}{'hi':>9}{'n_fill':>8}{'n_mkt':>7}  sig")
rows = []
for (b, a), g in fills.groupby(["px_bin", "action"], observed=True):
    m, lo, hi, nm = cluster_ci(g, "edge")
    sig = "YES" if (lo > 0 or hi < 0) else "."
    rows.append(dict(bucket=str(b), side=a, edge=m, lo=lo, hi=hi,
                     n_fill=len(g), n_mkt=nm, sig=sig))
    print(f"  {str(b):<12}{a:<6}{m:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
          f"{len(g):>8}{nm:>7}  {sig}")
edge_df = pd.DataFrame(rows)


# ===== 2. settlement edge vs +30s markout (adverse-selection check) =====
print("\n" + "=" * 84)
print("2.  SETTLEMENT edge vs +30s MID-MARKOUT, by side")
print("    (markout < 0 while settle-edge > 0  =>  edge is outcome luck, not price edge)")
print("=" * 84)
print(f"  {'side':<6}{'settle_edge':>13}{'markout_30s':>13}{'n_fill':>9}")
for a, g in fills.groupby("action"):
    se, _, _, _ = cluster_ci(g, "edge")
    mk_m, mk_lo, mk_hi, _ = cluster_ci(g.dropna(subset=["markout_30s"]), "markout_30s")
    print(f"  {a:<6}{se:>+13.4f}{mk_m:>+13.4f}{len(g):>9}   "
          f"markout CI[{mk_lo:+.4f},{mk_hi:+.4f}]")


# ===== 3. time-to-close split =====
print("\n" + "=" * 84)
print("3.  SETTLEMENT edge by TIME-TO-CLOSE bucket and side")
print("    (is the sell edge a late-window effect, like the HAR Brier edge?)")
print("=" * 84)
fills["ttc_bin"] = pd.cut(fills["ttc_s"], [0, 60, 120, 300, 900],
                          labels=["<1m", "1-2m", "2-5m", "5-15m"])
print(f"  {'ttc':<8}{'side':<6}{'edge':>9}{'lo':>9}{'hi':>9}{'n_fill':>8}  sig")
for (t, a), g in fills.groupby(["ttc_bin", "action"], observed=True):
    m, lo, hi, _ = cluster_ci(g, "edge")
    sig = "YES" if (lo > 0 or hi < 0) else "."
    print(f"  {str(t):<8}{a:<6}{m:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}{len(g):>8}  {sig}")


# ===== plot: settlement edge with clustered CI bars =====
fig, ax = plt.subplots(figsize=(12, 6))
labels = [str(b) for b in fills["px_bin"].cat.categories]
xpos = {l: i for i, l in enumerate(labels)}
for a, color, off in [("buy", "#ef4444", -0.12), ("sell", "#22c55e", 0.12)]:
    sub = edge_df[edge_df["side"] == a]
    x = [xpos[b] + off for b in sub["bucket"]]
    ax.errorbar(x, sub["edge"],
                yerr=[sub["edge"] - sub["lo"], sub["hi"] - sub["edge"]],
                fmt="o", color=color, capsize=3, markersize=7,
                label=a, alpha=0.85)
ax.axhline(0, color="#888", lw=1, ls="--")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_xlabel("price bucket")
ax.set_ylabel("settlement edge ($/contract, maker)  ·  ticker-clustered 95% CI")
ax.set_title("Edge by price bucket with clustered CIs — buy vs sell")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
out = Path(__file__).resolve().parent / "buckets_hardened_edge_ci.png"
fig.savefig(out, dpi=120)
print(f"\n[saved] {out}")
