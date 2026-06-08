"""Shared loader for deep-research scripts. Loads all available KXETH15M
data, derives per-fill P&L approximations (held-to-close + 5m markout),
and caches the result to parquet for fast re-runs."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0")
sys.path.insert(0, str(ROOT / "analysis"))
from utility import (
    load_all_data, theo_vec, theo_vec_twap, implied_sigma,
    calculate_markouts, brier_score, bootstrap_ci, ANN_MIN,
    realized_sigma_forward, SECONDS_PER_YEAR,
)

CACHE = ROOT / "analysis/Aston/AgentGenerated/_cache"
CACHE.mkdir(exist_ok=True)


def load(cutoff="26MAY15", force=False):
    """Returns dict with theo, book, spot, fills, events keyed DataFrames."""
    f = CACHE / f"all_{cutoff}.pkl"
    if f.exists() and not force:
        return pd.read_pickle(f)
    theo, book, spot, fills, events = load_all_data("KXETH15M", cutoff)
    out = dict(theo=theo, book=book, spot=spot, fills=fills, events=events)
    pd.to_pickle(out, f)
    return out


def attach_market_meta(fills, theo):
    """For each fill, attach the ticker's strike, close_time, and TWAP-derived
    settlement (yes/no). Returns fills with new columns."""
    g = theo.groupby("ticker").agg(
        strike=("strike", "first"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
        last_secs=("seconds_to_expiry", "min"),
    ).reset_index()
    g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
    return fills.merge(g[["ticker", "strike", "close_time"]], on="ticker", how="left", suffixes=("", "_meta"))


def compute_settlements_twap(theo, spot, window_s=60, min_ticks=5):
    """Same as utility.compute_settlements but returns dataframe with TWAP and outcome."""
    spot = spot.sort_values("ts")
    out = []
    for tk, g in theo.groupby("ticker"):
        last = g.loc[g["seconds_to_expiry"].idxmin()]
        secs_rem = float(last["seconds_to_expiry"])
        close_t = last["ts"] + pd.Timedelta(seconds=secs_rem)
        strike = float(last["strike"])
        w0 = close_t - pd.Timedelta(seconds=window_s)
        w = spot[(spot["ts"] >= w0) & (spot["ts"] <= close_t)]
        if len(w) < min_ticks:
            continue
        twap = float(w["price"].mean())
        out.append(dict(ticker=tk, strike=strike, close_time=close_t,
                        twap=twap, outcome=int(twap > strike)))
    return pd.DataFrame(out)


def pnl_held_to_close(fills, settlements):
    """P&L per fill if held to settlement. action='buy' means we bought YES.
    YES settles 1 if outcome=1; cost is fill price. P&L in cents."""
    f = fills.merge(settlements[["ticker", "outcome"]], on="ticker", how="inner")
    # cap fee at 0 for now (fee field is recorded but small)
    fee = f["fee"].fillna(0.0)
    payoff = np.where(f["action"] == "buy",
                      (f["outcome"] - f["price"]) * 100,
                      (f["price"] - f["outcome"]) * 100)
    f["pnl_settle_c"] = payoff * f["count"] - fee * 100
    return f


if __name__ == "__main__":
    d = load(force=True)
    for k, v in d.items():
        print(f"{k:8s} rows={len(v):,}")
    print()
    print("date range:", d["fills"]["ts"].min(), "to", d["fills"]["ts"].max())
    print("tickers in theo:", d["theo"]["ticker"].nunique())
    print("tickers with fills:", d["fills"]["ticker"].nunique())
