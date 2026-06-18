"""HAR-forecast sigma vs market-implied sigma, both scored against realized
sigma, bucketed by moneyness.

Question: does HAR's vol-forecast advantage hold in the OTM wings (where the
longshot binary actually prices off vega) or only near ATM / in the bulk?

For each market we pick a fixed evaluation instant (nearest theo row to a
target time-to-expiry). At that instant:
  - sigma_har   = theo_state.sigma          (HAR forecast)
  - sigma_imp   = implied_sigma(mid, ...)    (market's forecast)
  - sigma_real  = Parkinson sigma over [eval_ts, close]   (the truth)
Both forecasts predict realized over the SAME remaining window, so the
comparison is apples-to-apples. Moneyness = |ln(S/K)| / (sigma_har*sqrt(T))
~ |d2| at the eval point — the moneyness that drives the binary price.

Run from analysis/Aston/.  Reads recorder DBs (theo/book/spot); no API.

    python3 vol_accuracy_by_moneyness.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utility as U

START = "2026-05-16"
ANN = U.ANN_MIN
FOUR_LN2 = U.FOUR_LN2
# evaluation points: seconds-to-expiry to snap each market's forecast to.
EVAL_TARGETS = {"T~14min": 840, "T~7min": 420}
# moneyness (|d2|-ish) buckets
MN_EDGES = [0.0, 0.25, 0.75, 1.5, np.inf]
MN_LABELS = ["ATM (<0.25)", "near (0.25-0.75)", "mid-OTM (0.75-1.5)", "deep-OTM (>1.5)"]


def realized_sigma_window(spot, eval_ts, close_ts):
    """Annualized Parkinson sigma over [eval_ts, close_ts] from 1-min bars."""
    s = spot[(spot["ts"] >= eval_ts) & (spot["ts"] <= close_ts)]
    if len(s) < 3:
        return np.nan
    mins = s.assign(minute=s["ts"].dt.floor("1min")).groupby("minute")["price"]
    bars = mins.agg(["max", "min"])
    bars = bars[bars["min"] > 0]
    if bars.empty:
        return np.nan
    pv = np.log(bars["max"] / bars["min"]) ** 2 / FOUR_LN2
    pv = pv[bars["max"] > bars["min"]]
    n_min = max(len(bars), 1)
    var = pv.sum()
    return np.sqrt(max(var, 0.0) * (ANN / n_min))


def eval_at_target(theo, book, spot, target_secs):
    """One row per (ticker) snapped to the theo recompute nearest target_secs."""
    rows = []
    book = book.sort_values("ts")
    for ticker, tg in theo.groupby("ticker"):
        tg = tg.dropna(subset=["sigma", "spot", "strike", "seconds_to_expiry"])
        tg = tg[tg["seconds_to_expiry"] > 60]
        if tg.empty:
            continue
        i = (tg["seconds_to_expiry"] - target_secs).abs().idxmin()
        r = tg.loc[i]
        T_secs = float(r["seconds_to_expiry"])
        if not (target_secs * 0.5 <= T_secs <= target_secs * 1.6):
            continue
        close_ts = r["ts"] + pd.Timedelta(seconds=T_secs)
        # market mid at the eval instant (asof, backward)
        bk = book[book["ticker"] == ticker]
        mid = np.nan
        if not bk.empty:
            j = bk["ts"].searchsorted(r["ts"], side="right") - 1
            if j >= 0:
                mid = bk.iloc[j]["mid"]
        s_real = realized_sigma_window(spot[spot["ticker"] == ticker],
                                       r["ts"], close_ts)
        sig_har = float(r["sigma"])
        sig_imp = U.implied_sigma(mid, r["spot"], r["strike"], T_secs)
        if hasattr(sig_imp, "__len__"):
            sig_imp = float(np.asarray(sig_imp))
        T_yr = T_secs / (365.25 * 24 * 3600)
        u = sig_har * np.sqrt(T_yr)
        mny = abs(np.log(r["spot"] / r["strike"])) / u if u > 0 else np.nan
        rows.append({"ticker": ticker, "T_secs": T_secs, "mid": mid,
                     "sigma_har": sig_har, "sigma_imp": sig_imp,
                     "sigma_real": s_real, "moneyness": mny})
    df = pd.DataFrame(rows)
    return df.dropna(subset=["sigma_real", "sigma_imp", "moneyness"])


def score(df):
    df = df.copy()
    df["err_har"] = df["sigma_har"] - df["sigma_real"]
    df["err_imp"] = df["sigma_imp"] - df["sigma_real"]
    df["mn_bin"] = pd.cut(df["moneyness"], bins=MN_EDGES, labels=MN_LABELS)
    out = []
    for lbl, g in df.groupby("mn_bin", observed=True):
        out.append({
            "bucket": lbl, "n": len(g),
            "real_sigma": round(g["sigma_real"].mean(), 3),
            "har_bias": round(g["err_har"].mean(), 3),
            "imp_bias": round(g["err_imp"].mean(), 3),
            "har_mae": round(g["err_har"].abs().mean(), 3),
            "imp_mae": round(g["err_imp"].abs().mean(), 3),
            "har_corr": round(g["sigma_har"].corr(g["sigma_real"]), 3),
            "imp_corr": round(g["sigma_imp"].corr(g["sigma_real"]), 3),
        })
    res = pd.DataFrame(out).set_index("bucket").reindex(MN_LABELS).dropna(how="all")
    res["mae_edge(imp-har)"] = (res["imp_mae"] - res["har_mae"]).round(3)
    return res


def main():
    print(f"[load] theo/book/spot since {START} ...")
    theo = U.load_theo(START, until="today")
    book = U.load_book(START, until="today")
    # spot_ticks has no standalone loader; pull it the same way load_theo does
    from utility import _load_day_table, day_range, DEFAULT_LOCAL_DIR, \
        DEFAULT_S3_CACHE_DIR, DEFAULT_S3_BUCKET
    parts = [_load_day_table("KXETH15M", d, "spot_ticks", "ts,ticker,price",
                             None, None, DEFAULT_LOCAL_DIR, DEFAULT_S3_CACHE_DIR,
                             DEFAULT_S3_BUCKET)
             for d in day_range(START, "today")]
    spot = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    spot["ts"] = pd.to_datetime(spot["ts"], utc=True, format="ISO8601")
    print(f"[load] theo={len(theo):,} book={len(book):,} spot={len(spot):,}")

    pd.set_option("display.width", 170, "display.max_columns", 20)
    for name, tgt in EVAL_TARGETS.items():
        df = eval_at_target(theo, book, spot, tgt)
        print("\n" + "=" * 90)
        print(f"VOL ACCURACY BY MONEYNESS  @ {name}  (n={len(df)} markets)")
        print(f"forecast vs realized; lower MAE = better; "
              f"mae_edge>0 means HAR beats implied")
        print("=" * 90)
        print(score(df).to_string())


if __name__ == "__main__":
    main()
