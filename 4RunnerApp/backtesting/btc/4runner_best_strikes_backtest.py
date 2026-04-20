# btc_moneyness_analysis.py
import pandas as pd
import numpy as np

df = pd.read_csv("../data/backtest_btc_results.csv")

df["created_time"] = pd.to_datetime(df["created_time"])
df["expiration"] = pd.to_datetime(df["expiration"])
df["time_to_expiration"] = df["expiration"] - df["created_time"]

# config
quantity = 100
edge = 0.14
dte = 0.6

# how far is the strike from spot
df["moneyness"] = (df["strike"] - df["btc_price"]) / df["btc_price"]
df["moneyness_pct"] = (df["moneyness"] * 100).round(1)

# bucket moneyness into ranges
df["moneyness_bucket"] = pd.cut(df["moneyness_pct"],
    bins=[-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50],
    labels=["<-20%", "-20 to -10%", "-10 to -5%", "-5 to -2%", "-2 to 0%",
            "0 to 2%", "2 to 5%", "5 to 10%", "10 to 20%", ">20%"]
)

# settled only
df_settled = df.dropna(subset=["settle_val"])

# filter
filtered = df_settled[
    (df_settled["time_to_expiration"] > pd.Timedelta(days=dte)) &
    ((df_settled["theo"] - df_settled["market_price"]).abs() >= edge)
].copy()

filtered["theo_edge"] = filtered["theo"] + edge
filtered["theo_quantity"] = np.minimum(filtered["quantity"], quantity)
filtered["trade_pnl"] = np.where(
    filtered["settlement"] == "yes",
    (1 - filtered["theo_edge"]) * filtered["theo_quantity"],
    filtered["theo_edge"] * filtered["theo_quantity"]
)

# group by moneyness bucket
print(f"Edge: {edge} | Min DTE: {dte}d | Qty cap: {quantity}")
print(f"\n{'moneyness':<15} {'trades':<8} {'contracts':<10} {'pnl':<12} {'pnl/contract':<12} {'avg_theo':<10} {'avg_mkt':<10} {'win_rate':<10}")
print("-" * 90)

for bucket in filtered["moneyness_bucket"].cat.categories:
    sub = filtered[filtered["moneyness_bucket"] == bucket]
    if len(sub) == 0:
        continue
    trades = len(sub)
    contracts = int(sub["theo_quantity"].sum())
    pnl = sub["trade_pnl"].sum()
    pnl_per = pnl / contracts if contracts > 0 else 0
    avg_theo = sub["theo"].mean()
    avg_mkt = sub["market_price"].mean()
    wins = (sub["trade_pnl"] > 0).sum()
    win_rate = wins / trades if trades > 0 else 0

    print(f"{str(bucket):<15} {trades:<8} {contracts:<10} ${pnl:<11,.2f} ${pnl_per:<11.4f} {avg_theo:<10.4f} {avg_mkt:<10.4f} {win_rate:<10.1%}")

# also show by absolute distance from spot
print(f"\n\n--- By absolute distance from spot ---")
df_settled["abs_distance"] = abs(df_settled["strike"] - df_settled["btc_price"])
filtered["abs_distance"] = abs(filtered["strike"] - filtered["btc_price"])

# bucket by dollar distance
filtered["dist_bucket"] = pd.cut(filtered["abs_distance"],
    bins=[0, 1000, 2000, 5000, 10000, 20000, 50000],
    labels=["$0-1k", "$1k-2k", "$2k-5k", "$5k-10k", "$10k-20k", "$20k+"]
)

print(f"\n{'distance':<12} {'trades':<8} {'contracts':<10} {'pnl':<12} {'pnl/contract':<12} {'avg_theo':<10} {'avg_mkt':<10}")
print("-" * 75)

for bucket in filtered["dist_bucket"].cat.categories:
    sub = filtered[filtered["dist_bucket"] == bucket]
    if len(sub) == 0:
        continue
    trades = len(sub)
    contracts = int(sub["theo_quantity"].sum())
    pnl = sub["trade_pnl"].sum()
    pnl_per = pnl / contracts if contracts > 0 else 0
    avg_theo = sub["theo"].mean()
    avg_mkt = sub["market_price"].mean()

    print(f"{str(bucket):<12} {trades:<8} {contracts:<10} ${pnl:<11,.2f} ${pnl_per:<11.4f} {avg_theo:<10.4f} {avg_mkt:<10.4f}")