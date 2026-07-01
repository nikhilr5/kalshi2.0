"""Kalshi catalog screen: find RECURRING, RETAIL-dominated, LOW-CAPACITY series.

The thesis (where a solo operator can have edge): markets institutions skip on
capacity, that recur often enough to validate a pattern, and are dominated by
retail flow you can fade. This screen ranks the whole catalog on those axes.

  axis 1  recurring   -> series.frequency + #distinct events live now
  axis 2  active      -> sum 24h volume across the series' open markets
  axis 3  low-capacity-> liquidity / notional per market (institutions need size)
  axis 4  retail      -> median trade size on the series' busiest market (phase 2)

  python3 catalog_screen.py            # full screen
  python3 catalog_screen.py 40         # phase-2 trade-sample depth (default 30)
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
from kalshi_api import KalshiAPI            # noqa: E402

_f = lambda x: float(x) if x not in (None, "") else 0.0
RECUR = {"daily", "weekly"}                  # frequencies that give you sample fast


def pull_series(api):
    s = api._get("/series", {"limit": 100000}).get("series", [])
    rows = [dict(series=x["ticker"], category=x.get("category") or "?",
                 freq=x.get("frequency") or "?", title=x.get("title") or "",
                 fee_type=x.get("fee_type") or "?") for x in s]
    return pd.DataFrame(rows)


def pull_market_agg(api, series_list):
    """Targeted: pull OPEN markets per candidate series (skips the MVE/sports
    market flood entirely) and aggregate volume/OI/liquidity. One series-filtered
    /markets call each instead of paging the whole universe."""
    rows = []
    for i, s in enumerate(series_list):
        try:
            mk = api.get_markets(series_ticker=s, status="open")
        except Exception:
            mk = []
        if not mk:
            continue
        ev, v24, vtot, oi, liq, top_v, top_tk = set(), 0.0, 0.0, 0.0, 0.0, -1.0, None
        for m in mk:
            ev.add(m.get("event_ticker", ""))
            mv = _f(m.get("volume_24h_fp"))
            v24 += mv; vtot += _f(m.get("volume_fp"))
            oi += _f(m.get("open_interest_fp")); liq += _f(m.get("liquidity_dollars"))
            if mv > top_v:
                top_v, top_tk = mv, m.get("ticker")
        rows.append(dict(series=s, n_mkt=len(mk), n_event=len(ev), v24=v24,
                         vtot=vtot, oi=oi, liq=liq, top_tk=top_tk))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(series_list)} candidate series pulled")
    return pd.DataFrame(rows)


def trade_sample(api, ticker, limit=1000):
    """Median + p90 trade size (contracts) on one market -> retail proxy.
    Small median with a modest p90 = retail-dominated; big trades = institutional."""
    try:
        tr = api.get_trades(ticker, limit=limit)
    except Exception:
        return np.nan, np.nan, 0
    if not tr:
        return np.nan, np.nan, 0
    sizes = np.array([_f(t.get("count_fp")) for t in tr])
    return float(np.median(sizes)), float(np.percentile(sizes, 90)), len(sizes)


def main(depth=30):
    api = KalshiAPI()
    print("pulling series catalog...")
    ser = pull_series(api)
    print(f"  {len(ser)} series across {ser['category'].nunique()} categories")

    # ---- category-level landscape (counts, from catalog -- cheap) ----
    cat = (ser.groupby("category")
             .agg(series=("series", "count"),
                  recurring=("freq", lambda s: s.isin(RECUR).sum()))
             .sort_values("recurring", ascending=False))
    print("\n### CATEGORY LANDSCAPE (series count, # recurring daily/weekly) ###")
    print(cat.to_string())

    # ---- candidate filter BEFORE pulling volume: recurring, non-MVE ----
    cand = ser[ser["freq"].isin(RECUR) & ~ser["series"].str.startswith("KXMVE")].copy()
    print(f"\n{len(cand)} recurring (daily/weekly) non-MVE candidate series "
          f"-> pulling open-market volume for each...")
    mag = pull_market_agg(api, cand["series"].tolist())
    df = cand.merge(mag, on="series", how="left").fillna(
        dict(n_mkt=0, n_event=0, v24=0, vtot=0, oi=0, liq=0))

    # ---- the screen: active recurring candidates, retail signal via trade size ----
    act = df[(df["v24"] > 0) & (df["n_mkt"] > 0)].copy()
    act["oi_per_mkt"] = act["oi"] / act["n_mkt"].clip(lower=1)
    shortlist = act.sort_values("v24", ascending=False).head(depth).copy()
    print(f"\nphase 2: sampling trade size on busiest market of top {depth} recurring series...")
    med, p90, ntr = [], [], []
    for tk in shortlist["top_tk"]:
        a, b, n = (trade_sample(api, tk) if isinstance(tk, str) else (np.nan, np.nan, 0))
        med.append(a); p90.append(b); ntr.append(n)
    shortlist["med_trade"] = med
    shortlist["p90_trade"] = p90

    cols = ["series", "category", "freq", "n_event", "v24", "oi_per_mkt",
            "med_trade", "p90_trade", "title"]
    show = shortlist[cols].copy()
    show["title"] = show["title"].str.slice(0, 28)
    print("\n### SHORTLIST: recurring + active, ranked by 24h volume ###")
    print("  (med_trade small = retail-dominated; p90 small = no institutional size)")
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # retail cut: small median AND small p90 (no big players) -> our zone
    retail = shortlist[(shortlist["med_trade"] <= 25) & (shortlist["p90_trade"] <= 200)]
    retail = retail.sort_values("v24", ascending=False)
    print("\n### RETAIL CUT (med_trade<=25 AND p90<=200 contracts) ###")
    print(retail[["series", "category", "freq", "v24", "med_trade", "p90_trade", "title"]]
          .assign(title=lambda d: d["title"].str.slice(0, 28))
          .to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    df.to_csv("cache/catalog_screen.csv", index=False)
    print("\nfull catalog -> cache/catalog_screen.csv")
    return df, shortlist


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(d)
