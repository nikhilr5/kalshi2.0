"""HAR-RV sampling-frequency experiment: 1-min vs 5s Parkinson bars.

Builds two bar series from the SAME Coinbase ETH-USD ticks (1-min and 5s
H/L), fits HAR-RV on each (own-frequency label), then scores both feature
sets against a COMMON 5s label on identical OOS rows. Mirrors the
production pipeline in Aston/har_fit.py (Parkinson per-bar variance,
trailing 15m/30m/4h/24h RVs, non-overlapping 15-min steps, 80/20 temporal
split, OLS). Gaps (>5min no ticks) invalidate training rows whose 24h
lookback crosses them.

Offline research only. Rerunnable:
    python3 _har_sampling_freq.py
"""

import glob
import math
import os
import sqlite3

import numpy as np
import pandas as pd

DATA_DIRS = [
    "/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/_s3_cache",
    "/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data",
]
EXCLUDE_DAYS = {"26JUN09", "26JUN10"}  # partial / per-prompt corrupt

ANN_MINUTES = 365.25 * 24 * 60          # 525,960
ANN_SECONDS = ANN_MINUTES * 60
FOUR_LN2 = 4.0 * math.log(2.0)

# Horizons in MINUTES (wall-clock), same as production.
H_15M, H_30M, H_4H, H_24H = 15, 30, 240, 1440
STEP_MIN = 15
GAP_MIN = 5.0                            # >5min no ticks = data gap


def list_day_dbs():
    """Map YYMONDD -> path, preferring data/ over _s3_cache for dupes.
    Skips zero-byte (empty) DBs and excluded days."""
    found = {}
    for d in DATA_DIRS:
        for p in glob.glob(os.path.join(d, "KXETH15M-*.db")):
            if os.path.getsize(p) == 0:
                continue
            day = os.path.basename(p).replace("KXETH15M-", "").replace(".db", "")
            if day in EXCLUDE_DAYS:
                continue
            # data/ wins over cache only if cache absent; cache is canonical
            found.setdefault(day, p)
    return found


def load_ticks(path):
    con = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT ts, price FROM spot_ticks", con)
    con.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    return df.set_index("ts").sort_index()["price"]


def bars_from_ticks(price, freq):
    """Per-day H/L/count bars at pandas offset `freq` ('1min','5s')."""
    r = price.resample(freq)
    out = pd.DataFrame({"high": r.max(), "low": r.min(), "cnt": r.count()})
    return out


def build_bar_grid(days, freq):
    """Concat per-day bars onto one continuous wall-clock grid spanning
    min..max bar time. Empty bars (no ticks) carry cnt=0 and NaN H/L."""
    parts = []
    for day, path in sorted(days.items()):
        price = load_ticks(path)
        if price.empty:
            continue
        parts.append(bars_from_ticks(price, freq))
    bars = pd.concat(parts).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    full = pd.date_range(bars.index.min(), bars.index.max(), freq=freq, tz="UTC")
    bars = bars.reindex(full)
    bars["cnt"] = bars["cnt"].fillna(0)
    return bars


def parkinson_var(high, low):
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    out = np.zeros(len(h))
    safe = (h > 0) & (l > 0) & (h > l) & np.isfinite(h) & np.isfinite(l)
    out[safe] = np.log(h[safe] / l[safe]) ** 2 / FOUR_LN2
    return out


def annualize_var(var_sum, window_min):
    if var_sum <= 0:
        return 0.0
    return math.sqrt(var_sum * (ANN_MINUTES / window_min))


def gap_mask(bars, bar_min):
    """Per-bar gap flag: True where the bar had no ticks AND lies inside a
    stretch of >GAP_MIN minutes with no ticks. We mark any zero-tick bar;
    a run of consecutive zero-tick bars covering >GAP_MIN is a gap.
    Returns a bool array aligned to bars: True = inside a gap run."""
    empty = (bars["cnt"].to_numpy() == 0)
    flag = np.zeros(len(empty), dtype=bool)
    run_len = int(math.ceil(GAP_MIN / bar_min))
    i = 0
    n = len(empty)
    while i < n:
        if empty[i]:
            j = i
            while j < n and empty[j]:
                j += 1
            if (j - i) > run_len:        # strictly longer than 5 min
                flag[i:j] = True
            i = j
        else:
            i += 1
    return flag


def build_table(bars, bar_min, label_ann):
    """Non-overlapping 15-min-step training table on a bar grid.
    bar_min = bar length in minutes (1.0 or 5/60). label_ann lets us swap
    the OLS target sampling: 'own' uses this grid's label; a passed array
    overrides it (for common-label scoring).
    Returns X, y_own, valid_mask, and the row anchor timestamps."""
    var = parkinson_var(bars["high"], bars["low"])
    gap = gap_mask(bars, bar_min)
    n = len(var)
    bpm = 1.0 / bar_min                  # bars per minute
    w15 = int(round(H_15M * bpm))
    w30 = int(round(H_30M * bpm))
    w4h = int(round(H_4H * bpm))
    w24 = int(round(H_24H * bpm))
    step = int(round(STEP_MIN * bpm))

    cumvar = np.concatenate([[0.0], np.cumsum(var)])
    cumgap = np.concatenate([[0], np.cumsum(gap.astype(int))])

    def wsum(a, b):                      # sum var over [a,b)
        return cumvar[b] - cumvar[a]

    def gaps_in(a, b):
        return cumgap[b] - cumgap[a]

    X, y, valid, anchors = [], [], [], []
    idx = bars.index
    for t in range(w24, n - w15, step):
        # lookback spans [t-w24, t); label spans [t, t+w15)
        bad = (gaps_in(t - w24, t) > 0) or (gaps_in(t, t + w15) > 0)
        X.append([
            1.0,
            annualize_var(wsum(t - w15, t), H_15M),
            annualize_var(wsum(t - w30, t), H_30M),
            annualize_var(wsum(t - w4h, t), H_4H),
            annualize_var(wsum(t - w24, t), H_24H),
        ])
        y.append(annualize_var(wsum(t, t + w15), H_15M))
        valid.append(not bad)
        anchors.append(idx[t])
    return (np.array(X), np.array(y), np.array(valid),
            pd.DatetimeIndex(anchors))


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def r2(beta, X, y):
    pred = X @ beta
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def metrics(pred, y):
    err = pred - y
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 else float("nan")
    ss_res = (err ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2v = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return corr, mae, rmse, r2v


def vol_signature(days):
    """Mean annualized sigma at several sampling intervals."""
    intervals = [("1s", 1 / 60), ("5s", 5 / 60), ("10s", 10 / 60),
                 ("30s", 30 / 60), ("1m", 1.0), ("5m", 5.0)]
    rows = []
    for label, bar_min in intervals:
        freq = {"1s": "1s", "5s": "5s", "10s": "10s",
                "30s": "30s", "1m": "1min", "5m": "5min"}[label]
        sigmas = []
        for day, path in sorted(days.items()):
            price = load_ticks(path)
            if price.empty:
                continue
            b = bars_from_ticks(price, freq)
            var = parkinson_var(b["high"], b["low"])
            # per-bar annualized sigma, averaged over bars with ticks
            mask = b["cnt"].to_numpy() > 0
            v = var[mask]
            if len(v):
                sig = np.sqrt(v * (ANN_MINUTES / bar_min))
                sigmas.append(sig)
        allsig = np.concatenate(sigmas)
        rows.append((label, float(np.mean(allsig)), len(allsig)))
    return rows


def fmt_beta(b):
    return (f"b0={b[0]:+.4f} b15={b[1]:+.4f} b30={b[2]:+.4f} "
            f"b4h={b[3]:+.4f} b24h={b[4]:+.4f}")


def main():
    days = list_day_dbs()
    print(f"=== DATA INVENTORY ===")
    print(f"days found ({len(days)}): {','.join(sorted(days))}")

    print("\nBuilding 1-min bar grid...")
    bars_1m = build_bar_grid(days, "1min")
    print(f"  1-min bars: {len(bars_1m)} "
          f"({bars_1m.index.min()} -> {bars_1m.index.max()})")
    print("Building 5s bar grid...")
    bars_5s = build_bar_grid(days, "5s")
    print(f"  5s bars: {len(bars_5s)}")

    # Report gaps on the 1-min grid (coarser; gap detection identical).
    g = gap_mask(bars_1m, 1.0)
    print(f"  gap bars (1-min grid, runs >{GAP_MIN}min no ticks): "
          f"{int(g.sum())} of {len(g)}")

    # Build tables.
    X1, y1, v1, anc1 = build_table(bars_1m, 1.0, None)
    X5, y5, v5, anc5 = build_table(bars_5s, 5 / 60, None)
    print(f"\n1-min rows: {len(y1)} ({int(v1.sum())} valid after gap drop)")
    print(f"5s   rows: {len(y5)} ({int(v5.sum())} valid after gap drop)")

    # Keep only valid rows.
    X1, y1, anc1 = X1[v1], y1[v1], anc1[v1]
    X5, y5, anc5 = X5[v5], y5[v5], anc5[v5]

    # --- Own-label fits, 80/20 temporal split ---
    def split_fit(X, y):
        s = int(0.8 * len(y))
        beta = fit_ols(X[:s], y[:s])
        return beta, r2(beta, X[:s], y[:s]), r2(beta, X[s:], y[s:]), s
    b1, r2tr1, r2te1, s1 = split_fit(X1, y1)
    b5, r2tr5, r2te5, s5 = split_fit(X5, y5)

    print("\n=== COEFFICIENT TABLE (own-frequency label) ===")
    print(f"1-min: {fmt_beta(b1)}  R2_IS={r2tr1:.3f} R2_OOS={r2te1:.3f} "
          f"n_train={s1}")
    print(f"5s   : {fmt_beta(b5)}  R2_IS={r2tr5:.3f} R2_OOS={r2te5:.3f} "
          f"n_train={s5}")

    # --- COMMON-LABEL comparison ---
    # Align 1-min and 5s rows on shared anchor timestamps. Use the 5s label
    # as the common target (better estimate of latent vol). Score both
    # feature sets on the SAME OOS rows.
    df1 = pd.DataFrame(X1, index=anc1,
                       columns=["c", "f15", "f30", "f4h", "f24h"])
    df1["y1"] = y1
    df5 = pd.DataFrame(X5, index=anc5,
                       columns=["c", "f15", "f30", "f4h", "f24h"])
    df5["y5"] = y5
    common = df1.join(df5, how="inner", lsuffix="_1m", rsuffix="_5s")
    common = common.sort_index()
    print(f"\ncommon anchor rows: {len(common)}")

    Xc1 = common[["c_1m", "f15_1m", "f30_1m", "f4h_1m", "f24h_1m"]].to_numpy()
    Xc5 = common[["c_5s", "f15_5s", "f30_5s", "f4h_5s", "f24h_5s"]].to_numpy()
    yc = common["y5"].to_numpy()        # common label = 5s next-15m RV

    s = int(0.8 * len(yc))
    # Refit each feature set against the COMMON label on the train slice.
    bc1 = fit_ols(Xc1[:s], yc[:s])
    bc5 = fit_ols(Xc5[:s], yc[:s])
    pred1 = Xc1[s:] @ bc1
    pred5 = Xc5[s:] @ bc5

    print("\n=== COMMON-LABEL OOS (target = 5s next-15m RV, same rows) ===")
    for name, pred in [("1-min feats", pred1), ("5s feats", pred5)]:
        c, mae, rmse, r2v = metrics(pred, yc[s:])
        print(f"{name:12s}: corr={c:.3f} MAE={mae:.4f} RMSE={rmse:.4f} "
              f"R2={r2v:.3f}")
    print(f"(OOS n={len(yc)-s})")

    # --- Volatility signature ---
    print("\n=== VOLATILITY SIGNATURE (mean annualized sigma) ===")
    for label, msig, ncnt in vol_signature(days):
        print(f"  {label:4s}: mean_sigma={msig:.4f}  (n_bars={ncnt})")


if __name__ == "__main__":
    main()
