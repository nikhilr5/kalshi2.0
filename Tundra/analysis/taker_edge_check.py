"""Is there > taker-cost edge? Compare the (now sigma-calibrated) model's bucket
probability to the live market mid on every liquid weather book we've recorded,
net of the real taker cost = spread crossed + Kalshi fee 0.07*p*(1-p).

Snapshot study (~2 days of recorded books) -> distribution of |model-market| gap
in cents and the fraction that clears the hurdle. NOT a P&L backtest (we have no
historical prices) -- it answers 'are the disagreements big enough to be a taker'.
"""
import sqlite3, glob, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
SIG = json.load(open(HERE / "cache" / "sigma_model.json"))["stations"]
CITY_ST = {"NYC":"KNYC","LAX":"KLAX","OKC":"KOKC","BOS":"KBOS","DAL":"KDAL"}
SEASON = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}

def sigma_bias(station, lead, season):
    s = SIG.get(station)
    if not s: return None
    lead = str(min(max(int(lead),0),7))
    byls = s["by_lead_season"].get(lead, {})
    if season in byls and byls[season].get("sigma"):      # season-specific if enough n
        return byls[season]["sigma"], byls[season]["bias"]
    bl = s["by_lead"].get(lead)                            # else pooled-season
    return (bl["sigma"], bl["bias"]) if bl else None

def model_prob(lo, hi, mean, sigma):
    plo = 0.0 if lo <= -9e8 else norm.cdf(lo, mean, sigma)
    phi = 1.0 if hi >=  9e8 else norm.cdf(hi, mean, sigma)
    return float(phi - plo)

def taker_cost(price, half_spread):
    return half_spread + 0.07 * price * (1 - price)      # dollars

def main():
    rows = []
    for db in sorted(glob.glob(str(HERE/"data"/"WEATHER-*.db"))):
        con = sqlite3.connect(db)
        # latest book per ticker that is two-sided
        bk = pd.read_sql("""
            SELECT b.* FROM market_book b
            JOIN (SELECT ticker, MAX(ts) ts FROM market_book
                  WHERE yes_bid IS NOT NULL AND yes_ask IS NOT NULL GROUP BY ticker) m
            ON b.ticker=m.ticker AND b.ts=m.ts
            WHERE b.yes_bid IS NOT NULL AND b.yes_ask IS NOT NULL""", con)
        fc = pd.read_sql("SELECT city,event_day,forecast_high,MAX(ts) ts FROM forecasts "
                         "GROUP BY city,event_day", con)
        con.close()
        if bk.empty or fc.empty: continue
        f = {(r.city, pd.to_datetime(r.event_day).date()): r.forecast_high
             for r in fc.itertuples()}
        for r in bk.itertuples():
            st = CITY_ST.get(r.city)
            if st is None or r.bucket_lo is None: continue
            try:
                tgt = datetime.strptime(r.event_day, "%y%b%d").replace(tzinfo=timezone.utc)
                now = pd.to_datetime(r.ts, utc=True).to_pydatetime()
            except Exception: continue
            fh = f.get((r.city, tgt.date()))
            if fh is None: continue
            lead = (tgt.date() - now.date()).days
            sb = sigma_bias(st, lead, SEASON[tgt.month])
            if not sb: continue
            sigma, bias = sb
            p = model_prob(r.bucket_lo, r.bucket_hi, fh - bias, sigma)
            mid = (r.yes_bid + r.yes_ask) / 2
            spread = r.yes_ask - r.yes_bid
            if spread <= 0 or spread > 0.10: continue       # ignore junk/illiquid books
            gap = p - mid                                    # + => model thinks YES cheap
            cost = taker_cost(mid, spread/2)
            rows.append(dict(city=r.city, ticker=r.ticker, lead=lead, mid=mid,
                             spread=spread, model_p=p, gap=gap, agap=abs(gap),
                             cost=cost, net=abs(gap)-cost))
    d = pd.DataFrame(rows)
    if d.empty: print("no data"); return
    d.to_csv(HERE/"cache"/"taker_edge_snapshot.csv", index=False)
    print(f"liquid two-sided markets scored: {len(d)}  (spread<=10c)\n")
    print(f"median spread: {d.spread.median()*100:.1f}c | median taker cost: {d.cost.median()*100:.2f}c\n")
    print("=== |model - market| gap (cents) ===")
    for q in (.5,.75,.9):
        print(f"  {int(q*100)}th pct: {d.agap.quantile(q)*100:.1f}c")
    print(f"\nfrac with |gap| > 2c        : {(d.agap>.02).mean():.0%}")
    print(f"frac with |gap| > taker cost: {(d.net>0).mean():.0%}   <- clears the real hurdle")
    print(f"frac with net edge > 2c     : {(d.net>.02).mean():.0%}\n")
    print("=== by city: median gap, frac clearing hurdle, mean net when it clears ===")
    g = d.groupby("city").apply(lambda x: pd.Series({
        "n": len(x), "med_gap_c": round(x.agap.median()*100,1),
        "clears_%": round((x.net>0).mean()*100),
        "net_if_clears_c": round(x.loc[x.net>0,"net"].mean()*100,1) if (x.net>0).any() else 0,
    }), include_groups=False).reset_index()
    print(g.to_string(index=False))

if __name__ == "__main__":
    main()
