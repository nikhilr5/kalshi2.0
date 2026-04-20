# backtest.py
import pandas as pd
import numpy as np
from math import log, sqrt
from scipy.stats import norm
from datetime import datetime

print("Loading data...")
trades = pd.read_csv("data/kxgoldmon_trades.csv")
gold = pd.read_csv("data/gold_spot_5min.csv")
gvz = pd.read_csv("data/gvz_daily.csv")
settlements = pd.read_csv("data/settlements.csv")

# parse dates — all in UTC
trades["created_time"] = pd.to_datetime(trades["created_time"]).dt.tz_localize(None)
gold["datetime"] = pd.to_datetime(gold["datetime"]).dt.tz_localize(None)
gvz["date"] = pd.to_datetime(gvz["date"]).dt.tz_localize(None)

# sort
gold = gold.sort_values("datetime").reset_index(drop=True)
gvz = gvz.sort_values("date").reset_index(drop=True)
trades = trades.sort_values("created_time").reset_index(drop=True)

# ensure numeric
gold["close"] = pd.to_numeric(gold["close"], errors="coerce")
gvz["gvz_vol"] = pd.to_numeric(gvz["gvz_vol"], errors="coerce")
trades["yes_price"] = pd.to_numeric(trades["yes_price"], errors="coerce")

# build lookups
gold_lookup = gold[["datetime", "close"]].rename(columns={"close": "gold_price"})
gvz_lookup = gvz[["date", "gvz_vol"]].copy()

# build settlement lookup from fresh API data
settlement_lookup = {}
for _, row in settlements.iterrows():
    ticker = row["ticker"]
    yes_sub = str(row.get("yes_sub_title", ""))
    result = row.get("result", "")
    status = row.get("status", "")

    # determine contract type from yes_sub_title
    if "or above" in yes_sub.lower():
        contract_type = "above"
    elif "to" in yes_sub.lower():
        contract_type = "bracket"
    elif "or below" in yes_sub.lower() or "below" in yes_sub.lower():
        contract_type = "below"
    else:
        contract_type = "unknown"

    settlement_lookup[ticker] = {
        "result": result,
        "contract_type": contract_type,
        "status": status,
        "yes_sub_title": yes_sub,
    }


def parse_expiration(ticker):
    """Parse expiration datetime from ticker"""
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
        # convert to UTC: assume hour is ET, add 5 for EST or 4 for EDT
        # Feb/Mar = EST (+5), Apr+ = EDT (+4)
        if month < 3 or (month == 3 and day < 8):
            utc_hour = hour + 5
        else:
            utc_hour = hour + 4
        return datetime(year, month, day, utc_hour, 0)
    except:
        return None


def parse_strike(ticker):
    """Parse strike from ticker"""
    if "-T" in ticker:
        try:
            return float(ticker.split("-T")[1])
        except:
            return None
    elif "-B" in ticker:
        try:
            return float(ticker.split("-B")[1])
        except:
            return None
    return None


def n_d2(spot, strike, vol, days_to_exp, rate=0.05):
    """N(d2) = probability of finishing above strike"""
    if spot <= 0 or strike <= 0 or vol <= 0 or days_to_exp <= 0:
        return np.nan
    T = days_to_exp / 365.0
    d2 = (log(spot / strike) + (rate - 0.5 * vol**2) * T) / (vol * sqrt(T))
    return norm.cdf(d2)


# build expiration lookup
ticker_expiration = {}
for ticker in trades["ticker"].unique():
    ticker_expiration[ticker] = parse_expiration(ticker)

# process each trade
print("Computing theos...")
results = []
total = len(trades)

for i, trade in trades.iterrows():
    if i % 500 == 0:
        print(f"  {i}/{total}...")

    ticker = trade["ticker"]
    trade_time = trade["created_time"]
    yes_price = trade["yes_price"]
    expiration = ticker_expiration.get(ticker)
    strike = parse_strike(ticker)
    info = settlement_lookup.get(ticker, {})
    contract_type = info.get("contract_type", "unknown")

    if expiration is None or pd.isna(yes_price) or strike is None:
        continue
    if contract_type == "unknown":
        continue

    # find closest gold price (look back)
    mask = gold_lookup["datetime"] <= trade_time
    if mask.any():
        idx = gold_lookup.loc[mask, "datetime"].idxmax()
        gold_price = gold_lookup.loc[idx, "gold_price"]
    else:
        continue

    # find gvz for that day
    gvz_mask = gvz_lookup["date"] <= trade_time
    if gvz_mask.any():
        gvz_idx = gvz_lookup.loc[gvz_mask, "date"].idxmax()
        vol = gvz_lookup.loc[gvz_idx, "gvz_vol"]
    else:
        continue

    # days to expiration
    days_to_exp = (expiration - trade_time).total_seconds() / 86400.0
    if days_to_exp <= 0:
        continue

    # compute theo based on contract type
    if contract_type == "above":
        # T contracts: "X or above" — yes pays if gold >= strike
        # theo = N(d2) = prob above
        theo = n_d2(gold_price, strike, vol, days_to_exp)
        market_price = yes_price

    elif contract_type == "below":
        # yes pays if gold < strike
        # theo = 1 - N(d2) = prob below
        prob_above = n_d2(gold_price, strike, vol, days_to_exp)
        if np.isnan(prob_above):
            continue
        theo = 1.0 - prob_above
        market_price = yes_price

    elif contract_type == "bracket":
        # yes pays if gold in range [lower, upper)
        lower = strike - 20.0
        upper = strike + 20.0
        prob_above_lower = n_d2(gold_price, lower, vol, days_to_exp)
        prob_above_upper = n_d2(gold_price, upper, vol, days_to_exp)
        if np.isnan(prob_above_lower) or np.isnan(prob_above_upper):
            continue
        theo = prob_above_lower - prob_above_upper
        market_price = yes_price

    else:
        continue

    if np.isnan(theo):
        continue

    # settlement: use fresh API data, yes=1, no=0
    result = info.get("result", "")
    if result == "yes":
        settle_val = 1.0
    elif result == "no":
        settle_val = 0.0
    else:
        settle_val = np.nan

    edge = theo - market_price
    pnl = settle_val - market_price if not np.isnan(settle_val) else np.nan

    quantity = trade.get("quantity", 1)
    if pd.isna(quantity):
        quantity = 1

    results.append({
        "ticker": ticker,
        "strike": strike,
        "contract_type": contract_type,
        "yes_sub_title": info.get("yes_sub_title", ""),
        "created_time": trade_time,
        "expiration": expiration,
        "days_to_exp": round(days_to_exp, 2),
        "gold_price": gold_price,
        "gvz_vol": round(vol, 4),
        "market_price": round(market_price, 4),
        "quantity": round(quantity, 3),
        "theo": round(theo, 4),
        "edge": round(edge, 4),
        "settlement": result,
        "settle_val": settle_val,
        "pnl": round(pnl, 4) if not np.isnan(pnl) else np.nan,
    })

df = pd.DataFrame(results)
df.to_csv("data/backtest_results.csv", index=False)

print(f"\nTotal results: {len(df)}")
print(f"Above contracts: {(df['contract_type'] == 'above').sum()}")
print(f"Below contracts: {(df['contract_type'] == 'below').sum()}")
print(f"Bracket contracts: {(df['contract_type'] == 'bracket').sum()}")
print(f"Settled trades: {df['settle_val'].notna().sum()}")

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

    for ct in ["above", "below", "bracket"]:
        sub = settled[settled["contract_type"] == ct]
        if len(sub) > 0:
            print(f"\n{ct.upper()} contracts:")
            print(f"  Count: {len(sub)} | Mean edge: {sub['edge'].mean():.4f} | Mean pnl: {sub['pnl'].mean():.4f}")

print(f"\nSaved to data/backtest_results.csv")