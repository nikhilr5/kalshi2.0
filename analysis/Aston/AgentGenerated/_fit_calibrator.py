"""Fit an isotonic recalibrator on the existing settled markets and show
what the function looks like. Pure analysis — does not modify Aston."""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/analysis")

def load():
    return pd.read_pickle("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache/all_26MAY15.pkl")

# Pool-adjacent-violators (PAV) — closed-form isotonic regression, no sklearn needed.
def fit_isotonic(x, y):
    """Returns sorted_x, fitted_y arrays defining a step function via linear interp."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    # PAV algorithm
    n = len(ys)
    vals = ys.copy()
    weights = np.ones(n)
    starts = np.arange(n)
    i = 0
    while i < n - 1:
        if vals[i] <= vals[i+1] + 1e-12:
            i += 1
            continue
        # Pool i and i+1
        new_w = weights[i] + weights[i+1]
        new_v = (vals[i]*weights[i] + vals[i+1]*weights[i+1]) / new_w
        vals[i] = new_v
        weights[i] = new_w
        # Remove i+1 by shifting
        vals = np.delete(vals, i+1)
        weights = np.delete(weights, i+1)
        starts = np.delete(starts, i+1)
        n -= 1
        if i > 0:
            i -= 1
    # Expand back into per-input fitted values
    fitted = np.zeros(len(xs))
    out_idx = 0
    for s_idx in range(len(starts)):
        end = starts[s_idx+1] if s_idx+1 < len(starts) else len(xs)
        fitted[starts[s_idx]:end] = vals[s_idx]
    return xs, fitted

def predict_isotonic(xs_train, ys_train, x_new):
    """Step-function interpolation, clipped to [0, 1]."""
    return np.clip(np.interp(x_new, xs_train, ys_train), 0.0, 1.0)

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
setts = pd.read_pickle(ROOT / "settlements.pkl")
d = load()
theo = d["theo"]

# Pull every theo snapshot for settled tickers in the trading regime
# (T-30s to T-12m) — that's where Aston is actually pricing.
g = theo.groupby("ticker").agg(
    last_ts=("ts","max"), last_secs=("seconds_to_expiry","min"),
).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(setts[["ticker","outcome"]], on="ticker", how="inner")

t = theo.merge(g[["ticker","close_time","outcome"]], on="ticker", how="inner")
t["secs_to_close"] = (t["close_time"] - t["ts"]).dt.total_seconds()
# Trading regime: 30s to 12m
t = t[(t["secs_to_close"] >= 30) & (t["secs_to_close"] <= 720)]
# Down-sample to one snapshot per 5s per ticker to avoid one ticker dominating
t["bucket"] = (t["secs_to_close"] // 5).astype(int)
t = t.sort_values(["ticker","bucket","secs_to_close"]).drop_duplicates(["ticker","bucket"], keep="first")

X = t["theo"].clip(0, 1).values
y = t["outcome"].astype(float).values
print(f"training pairs: {len(X):,}")

# Fit isotonic (PAV)
xs_cal, ys_cal = fit_isotonic(X, y)
class _Cal:
    def predict(self, x): return predict_isotonic(xs_cal, ys_cal, np.asarray(x))
cal = _Cal()

# Show the function as a table
print("\n" + "=" * 64)
print("RECALIBRATION FUNCTION  (raw_theo  →  recalibrated_theo)")
print("=" * 64)
grid = np.arange(0.0, 1.01, 0.05)
pred = cal.predict(grid)
print(f"  {'raw':>8}  {'recal':>8}  {'shift':>8}  {'interpretation':<30}")
for r, p in zip(grid, pred):
    shift = p - r
    arrow = "→" if abs(shift) < 0.005 else ("↓" if shift < 0 else "↑")
    label = ("(unchanged)" if abs(shift) < 0.01
            else f"({'overestimate' if shift < 0 else 'underestimate'})")
    print(f"  {r:>8.2f}  {p:>8.3f}  {shift:>+8.3f}  {arrow} {label}")

# Diagnostic on a finer grid
print("\n--- Same curve, finer grid in the bias zone (0.10–0.60) ---")
grid = np.arange(0.10, 0.61, 0.025)
pred = cal.predict(grid)
print(f"  {'raw':>8}  {'recal':>8}  {'shift_c':>8}")
for r, p in zip(grid, pred):
    print(f"  {r:>8.3f}  {p:>8.3f}  {(p-r)*100:>+7.1f}c")

# Brier comparison — does it actually improve out-of-sample?
print("\n" + "=" * 64)
print("OUT-OF-SAMPLE TEST  (last 25% time-ordered as holdout)")
print("=" * 64)
t_sorted = t.sort_values("ts").reset_index(drop=True)
split = int(len(t_sorted) * 0.75)
tr, te = t_sorted.iloc[:split], t_sorted.iloc[split:]
print(f"  train n={len(tr):,}  ({tr['ts'].min():%Y-%m-%d} → {tr['ts'].max():%Y-%m-%d})")
print(f"  test  n={len(te):,}  ({te['ts'].min():%Y-%m-%d} → {te['ts'].max():%Y-%m-%d})")

xs_tr, ys_tr = fit_isotonic(tr["theo"].clip(0,1).values, tr["outcome"].astype(float).values)
raw_te = te["theo"].clip(0,1).values
recal_te = predict_isotonic(xs_tr, ys_tr, raw_te)
out_te = te["outcome"].astype(float).values

b_raw = ((raw_te - out_te) ** 2).mean()
b_recal = ((recal_te - out_te) ** 2).mean()
print(f"\n  Brier raw theo:   {b_raw:.4f}")
print(f"  Brier recal theo: {b_recal:.4f}")
print(f"  Improvement:      {b_raw - b_recal:+.4f}  ({100*(b_raw-b_recal)/b_raw:+.1f}%)")

# Also: how much would buy-side P&L have improved if we used recal_theo
# for the quoting decision?  We need fills for that.
print("\n" + "=" * 64)
print("WOULD-BE BUY-SIDE BLEED REDUCTION  (counterfactual on actual fills)")
print("=" * 64)
f = pd.read_pickle(ROOT / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()].dropna(
    subset=["theo","outcome","pnl_settle_c"])
# fit on EVERYTHING (the question is "what does the lookup say")
xs_full, ys_full = fit_isotonic(X, y)
f["theo_recal"] = predict_isotonic(xs_full, ys_full, f["theo"].clip(0,1).values)

# At post time, our buy price = theo_recal - edge_bid, sell = theo_recal + edge_ask
# Approximate: with recal_theo, would we have posted at this fill price?
# Buy fill happens at price P; our buy quote was theo - 7c. The COUNTERPARTY
# crossed our quote so they would also cross theo_recal - 7c (since recal < theo
# in bias zone, our recal quote is even LOWER, i.e. less aggressive, so we wouldn't
# have been hit). Drop buys where theo_recal - 7c < fill_price (we'd have been below).
EDGE_BID = 0.07
EDGE_ASK = 0.05
f["recal_buy_post"] = f["theo_recal"] - EDGE_BID
f["recal_sell_post"] = f["theo_recal"] + EDGE_ASK
# Buy: kept if our recal buy quote >= fill price (we'd still have been at/above)
# Sell: kept if our recal sell quote <= fill price
keep_buy = (f["action"]=="buy") & (f["recal_buy_post"] >= f["price"])
keep_sell = (f["action"]=="sell") & (f["recal_sell_post"] <= f["price"])
keep = keep_buy | keep_sell

dropped = f[~keep & f["action"].isin(["buy","sell"])]
kept = f[keep]
n_days = f["date"].nunique()

print(f"  Baseline:                ${f['pnl_settle_c'].sum()/100:+.2f}  "
      f"per_day=${f['pnl_settle_c'].sum()/100/n_days:+.2f}")
print(f"  With recalibrated theo:  ${kept['pnl_settle_c'].sum()/100:+.2f}  "
      f"per_day=${kept['pnl_settle_c'].sum()/100/n_days:+.2f}")
print(f"  Δ vs baseline:           ${(kept['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:+.2f}/day")
print(f"  Fills dropped: {len(dropped):,}  ({100*len(dropped)/len(f):.1f}%)")
for action in ["buy","sell"]:
    sub_dr = dropped[dropped["action"]==action]
    sub_kp = kept[kept["action"]==action]
    print(f"    {action}: dropped n={len(sub_dr):>4} pnl=${sub_dr['pnl_settle_c'].sum()/100:+.2f}  "
          f"kept n={len(sub_kp):>4} pnl=${sub_kp['pnl_settle_c'].sum()/100:+.2f}")
