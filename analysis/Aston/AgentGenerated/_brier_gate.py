"""Verify: at z<0, when theo > market (we'd buy), who's right?
If market is consistently right at z<0, that's a structural fix:
gate our buys on "theo agrees with market direction".
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load

d = load()
theo = d["theo"]; book = d["book"]
ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
setts = pd.read_pickle(ROOT / "settlements.pkl")
from utility import SECONDS_PER_YEAR, bootstrap_ci

# T-5m snapshot
g = theo.groupby("ticker").agg(
    last_ts=("ts", "max"), last_secs=("seconds_to_expiry", "min"),
    strike=("strike", "first"),
).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(setts[["ticker", "outcome"]], on="ticker", how="inner")
theo_s = theo.merge(g[["ticker", "close_time", "outcome"]], on="ticker", how="inner")
theo_s["secs_to_close"] = (theo_s["close_time"] - theo_s["ts"]).dt.total_seconds()

snaps = theo_s[theo_s["secs_to_close"].between(295, 305)]
snaps = snaps.sort_values(["ticker", "secs_to_close"]).drop_duplicates("ticker", keep="first").copy()

bk = book[["ts", "ticker", "yes_bid", "yes_ask"]].sort_values("ts")
snaps = snaps.sort_values("ts")
snaps = pd.merge_asof(snaps, bk, by="ticker", on="ts", direction="backward",
                     tolerance=pd.Timedelta("3s"))
snaps["mid"] = (snaps["yes_bid"] + snaps["yes_ask"]) / 2
snaps = snaps.dropna(subset=["mid", "theo", "outcome"])
T = snaps["seconds_to_expiry"] / SECONDS_PER_YEAR
snaps["z"] = np.log(snaps["spot"] / snaps["strike"]) / (snaps["sigma"] * np.sqrt(T))
snaps["disagree"] = snaps["theo"] - snaps["mid"]
print(f"n snapshots T-5m: {len(snaps)}")

print("\n--- Who's right when theo and mid disagree, by z bucket? ---")
snaps["z_bucket"] = pd.cut(snaps["z"], bins=[-100, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 100])
print(f"{'z_bucket':>16} {'n':>4} {'B_theo':>9} {'B_mid':>9} {'theo_better':>12}")
for zb, gg in snaps.groupby("z_bucket", observed=True):
    if len(gg) < 5: continue
    bt = ((gg["theo"] - gg["outcome"])**2).mean()
    bm = ((gg["mid"]  - gg["outcome"])**2).mean()
    label = "+" if bt < bm else "-"
    diff = ((gg["theo"]-gg["outcome"])**2 - (gg["mid"]-gg["outcome"])**2).values
    lo, hi = bootstrap_ci(diff, B=2000)
    print(f"  {str(zb):>16} {len(gg):>4} {bt:>9.4f} {bm:>9.4f} {label}{abs(bt-bm):>8.4f}  CI[{lo:+.4f},{hi:+.4f}]")

# More fine-grained: theo>mid, theo<mid, by z
print("\n--- Disagreement direction: who wins? ---")
print("'theo>mid' means we'd buy; 'theo<mid' means we'd sell.")
snaps["case"] = np.where(snaps["theo"] > snaps["mid"], "theo>mid", "theo<mid")
for case in ["theo>mid", "theo<mid"]:
    sub = snaps[snaps["case"] == case]
    bt = ((sub["theo"] - sub["outcome"])**2).mean()
    bm = ((sub["mid"]  - sub["outcome"])**2).mean()
    print(f"  {case}:  n={len(sub):>4}  B_theo={bt:.4f}  B_mid={bm:.4f}  "
          f"{'theo better' if bt<bm else 'mid better'}")

# Now per zone
print("\n--- theo>mid (we'd buy) by z bucket ---")
print(f"{'z':>16} {'n':>4} {'B_theo':>8} {'B_mid':>8} {'theo_avg':>9} {'mid_avg':>9} {'actual':>9}")
b = snaps[snaps["case"] == "theo>mid"]
for zb, gg in b.groupby("z_bucket", observed=True):
    if len(gg) < 5: continue
    bt = ((gg["theo"] - gg["outcome"])**2).mean()
    bm = ((gg["mid"]  - gg["outcome"])**2).mean()
    print(f"  {str(zb):>16} {len(gg):>4} {bt:>8.4f} {bm:>8.4f} "
          f"{gg['theo'].mean():>9.4f} {gg['mid'].mean():>9.4f} {gg['outcome'].mean():>9.4f}")

print("\n--- theo<mid (we'd sell) by z bucket ---")
b = snaps[snaps["case"] == "theo<mid"]
for zb, gg in b.groupby("z_bucket", observed=True):
    if len(gg) < 5: continue
    bt = ((gg["theo"] - gg["outcome"])**2).mean()
    bm = ((gg["mid"]  - gg["outcome"])**2).mean()
    print(f"  {str(zb):>16} {len(gg):>4} {bt:>8.4f} {bm:>8.4f} "
          f"{gg['theo'].mean():>9.4f} {gg['mid'].mean():>9.4f} {gg['outcome'].mean():>9.4f}")

# Now the EDGE-vs-actual P&L of using theo vs mid as our pricing tool
# Per snapshot: if theo>mid, our hypothetical "buy at mid" P&L = outcome-mid (we paid mid)
# If theo<mid, our hypothetical "sell at mid" P&L = mid-outcome
snaps["if_we_acted"] = np.where(snaps["theo"]>snaps["mid"],
                                 snaps["outcome"]-snaps["mid"],
                                 snaps["mid"]-snaps["outcome"])
print("\n--- 'If we acted on theo (taking) at mid' — per-snapshot P&L by z ---")
print(f"{'z':>16} {'n':>4} {'mean_pnl_c':>12}")
for zb, gg in snaps.groupby("z_bucket", observed=True):
    if len(gg) < 5: continue
    print(f"  {str(zb):>16} {len(gg):>4} {(gg['if_we_acted']*100).mean():>+11.2f}c")
