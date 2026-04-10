# btc_strike_pnl.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("../data/backtest_btc_results.csv")

df["created_time"] = pd.to_datetime(df["created_time"])
df["expiration"] = pd.to_datetime(df["expiration"])
df["time_to_expiration"] = df["expiration"] - df["created_time"]

# config
quantity = 100
edge = .14
dte = .6

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

# group by strike
grouped = filtered.groupby("strike").agg(
    trades=("strike", "count"),
    total_qty=("theo_quantity", "sum"),
    total_pnl=("trade_pnl", "sum"),
    avg_theo=("theo", "mean"),
    avg_market=("market_price", "mean"),
).reset_index()

grouped = grouped.sort_values("strike")

print(f"\nEdge: {edge} | Min DTE: {dte}d | Qty cap: {quantity}")
print(f"Total tickers: {len(grouped)} | Total trades: {int(grouped['trades'].sum())} | Total PnL: ${grouped['total_pnl'].sum():,.2f}")

# graph
fig, ax = plt.subplots(figsize=(16, 8))

colors = ["#22c55e" if p >= 0 else "#ef4444" for p in grouped["total_pnl"]]
sizes = grouped["trades"] * 3 + 20

scatter = ax.scatter(grouped["strike"], grouped["total_pnl"], c=colors, s=sizes, alpha=0.7, edgecolors="white", linewidth=0.5)

# label each point with trade count
for _, r in grouped.iterrows():
    pnl = r["total_pnl"]
    offset = 10 if pnl >= 0 else -15
    ax.annotate(
        f"{int(r['trades'])} trades\n${pnl:,.0f}",
        (r["strike"], pnl),
        textcoords="offset points",
        xytext=(0, offset),
        ha="center",
        fontsize=6,
        color="#22c55e" if pnl >= 0 else "#ef4444",
        fontweight="bold",
    )

ax.axhline(y=0, color="gray", linewidth=1)
ax.set_xlabel("Strike ($)", fontsize=12)
ax.set_ylabel("Total PnL ($)", fontsize=12)
ax.set_title(f"BTC Weekly Bracket | PnL by Strike\nedge: {edge} | min DTE: {dte}d | qty cap: {quantity}", fontsize=13)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.grid(True, alpha=0.2)

# size legend
for s, label in [(20, "1 trade"), (50, "10 trades"), (110, "30 trades")]:
    ax.scatter([], [], s=s, c="gray", alpha=0.5, edgecolors="white", label=label)
ax.legend(loc="upper right", fontsize=9, title="Bubble size = trades")

fig.tight_layout()
filename = f"./graphs/btc_strike_pnl_e{edge}_d{dte}.png"
fig.savefig(filename, dpi=150)
plt.close(fig)
print(f"\nSaved to {filename}")