# backtest_btc.py
import pandas as pd
import numpy as np
from math import log, sqrt
from scipy.stats import norm
from datetime import datetime

print("Loading data...")
trades = pd.read_csv("../data/kxbtc_weekly_bracket_trades.csv")
btc = pd.read_csv("../data/btc_spot_5min.csv")

# parse dates — all UTC
trades["created_time"] = pd.to_datetime(trades["created_time"]).dt.tz_localize(None)
btc["datetime"] = pd.to_datetime(btc["datetime"]).dt.tz_localize(None)

# sort
btc = btc.sort_values("datetime").reset_index(drop=True)
trades = trades.sort_values("created_time").reset_index(drop=True)

# ensure numeric
btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
trades["yes_price"] = pd.to_numeric(trades["yes_price"], errors="coerce")
trades["strike"] = pd.to_numeric(trades["strike"], errors="coerce")

# build btc price lookup
btc_lookup = btc[["datetime", "close"]].rename(columns={"close": "btc_price"})

# fixed vol and rate
VOL = 0.60
RATE = 0.0001


def parse_expiration(ticker):
    """Parse expiration from ticker — e.g. KXBTC-26APR1017 = Apr 10 2026 5pm EDT"""
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    date_str = parts[1]
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    try:
        year = 2000 + int(date_str[:2])
        month = month_map[date_str[2:5]]
        day = int(date_str[5:7])
        hour = int(date_str[7:9]) if len(date_str) >= 9 else 17
        # convert EDT to UTC: +4 hours
        utc_hour = hour + 4
        return datetime(year, month, day, utc_hour, 0)
    except:
        return None


def n_d2(spot, strike, vol, days_to_exp, rate=RATE):
    """N(d2) = probability of finishing above strike"""
    if spot <= 0 or strike <= 0 or vol <= 0 or days_to_exp <= 0:
        return np.nan
    T = days_to_exp / 365.0
    d2 = (log(spot / strike) + (rate - 0.5 * vol**2) * T) / (vol * sqrt(T))
    return norm.cdf(d2)


def calc_bracket_theo(spot, strike, vol, days_to_exp):
    """Bracket theo: prob of landing in [strike-250, strike+250)"""
    lower = strike - 250.0
    upper = strike + 250.0
    prob_above_lower = n_d2(spot, lower, vol, days_to_exp)
    prob_above_upper = n_d2(spot, upper, vol, days_to_exp)
    if np.isnan(prob_above_lower) or np.isnan(prob_above_upper):
        return np.nan
    return prob_above_lower - prob_above_upper


# build expiration lookup
ticker_expiration = {}
for ticker in trades["ticker"].unique():
    ticker_expiration[ticker] = parse_expiration(ticker)

# process each trade
print("Computing theos...")
results = []
total = len(trades)

for i, trade in trades.iterrows():
    if i % 5000 == 0:
        print(f"  {i}/{total}...")

    ticker = trade["ticker"]
    trade_time = trade["created_time"]
    yes_price = trade["yes_price"]
    strike = trade["strike"]
    expiration = ticker_expiration.get(ticker)

    if expiration is None or pd.isna(yes_price) or pd.isna(strike):
        continue

    # find closest btc price (look back)
    mask = btc_lookup["datetime"] <= trade_time
    if mask.any():
        idx = btc_lookup.loc[mask, "datetime"].idxmax()
        btc_price = btc_lookup.loc[idx, "btc_price"]
    else:
        continue

    # days to expiration
    days_to_exp = (expiration - trade_time).total_seconds() / 86400.0
    if days_to_exp <= 0:
        continue

    # compute bracket theo
    theo = calc_bracket_theo(btc_price, strike, VOL, days_to_exp)
    if np.isnan(theo):
        continue

    # settlement
    settlement = trade["settlement_result"]
    if settlement == "yes":
        settle_val = 1.0
    elif settlement == "no":
        settle_val = 0.0
    else:
        settle_val = np.nan

    market_price = yes_price
    edge = theo - market_price
    pnl = settle_val - market_price if not np.isnan(settle_val) else np.nan


    quantity = trade.get("quantity", 1)
    if pd.isna(quantity):
        quantity = 1

    results.append({
        "ticker": ticker,
        "strike": strike,
        "contract_type": "bracket",
        "yes_sub_title": trade["yes_sub_title"],
        "created_time": trade_time,
        "expiration": expiration,
        "days_to_exp": round(days_to_exp, 2),
        "btc_price": btc_price,
        "vol": VOL,
        "market_price": round(market_price, 4),
        "quantity": round(quantity, 4),
        "theo": round(theo, 4),
        "edge": round(edge, 4),
        "settlement": settlement,
        "settle_val": settle_val,
        "pnl": round(pnl, 4) if not np.isnan(pnl) else np.nan,
        "expiration_value": trade["expiration_value"],
    })

df = pd.DataFrame(results)
df.to_csv("../data/backtest_btc_results.csv", index=False)

print(f"\nTotal results: {len(df)}")
print(f"Settled trades: {df['settle_val'].notna().sum()}")
print(f"Unsettled trades: {df['settle_val'].isna().sum()}")

print(f"\nEdge stats:")
print(df["edge"].describe())

settled = df.dropna(subset=["pnl"])
print(f"\nPnL stats (settled only):")
print(settled["pnl"].describe())

print(f"\nMean edge: {df['edge'].mean():.4f}")
print(f"Mean pnl:  {settled['pnl'].mean():.4f}")

# signal check
if len(settled) > 10:
    positive_edge = settled[settled["edge"] > 0]
    negative_edge = settled[settled["edge"] <= 0]
    print(f"\nPositive edge trades: {len(positive_edge)} | avg pnl: {positive_edge['pnl'].mean():.4f}")
    print(f"Negative edge trades: {len(negative_edge)} | avg pnl: {negative_edge['pnl'].mean():.4f}")

# verify a few theos against settlement
print("\nSample settled trades:")
sample = settled.head(10)
print(sample[["ticker", "strike", "btc_price", "theo", "market_price", "edge", "settlement", "expiration_value"]].to_string())

print(f"\nSaved to ../data/backtest_btc_results.csv")