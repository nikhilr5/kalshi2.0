import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/backtest_results.csv")


edge = 0.1
quantity = 100
daysTilExpiration = 0.1

df["created_time"] = pd.to_datetime(df["created_time"])
df["expiration"] = pd.to_datetime(df["expiration"])

df["time_to_expiration"] = df["expiration"] - df["created_time"]

#theorically where I would have placed my order at
df["theo_edge"] = df["theo"] + edge

#theo quantity i would have used
df["theo_quantity"] = np.minimum(df["quantity"], quantity)

bracket_high = df[
    (df["contract_type"] == "bracket") &
    (df["market_price"] > 0.1)
]

bracket_high_yes = df[
    (df["contract_type"] == "bracket") &
    (df["market_price"] > 0.1) &
    (df["settlement"] == 'yes')
]

print(df.columns)

# settled only
df_settled = bracket_high.dropna(subset=["settle_val"])

# YES trades
yes_trades = df_settled[df_settled["settlement"] == "yes"]
sum_yes = ((1 - yes_trades["market_price"]) * yes_trades["quantity"]).sum()

# NO trades
no_trades = df_settled[df_settled["settlement"] == "no"]
sum_no = (no_trades["market_price"] * no_trades["quantity"]) .sum()


yes_filtered = yes_trades[
    (yes_trades["time_to_expiration"] > pd.Timedelta(days=daysTilExpiration)) &
    ((yes_trades["theo"] - yes_trades["market_price"]).abs() >= edge)
]


no_filtered = no_trades[
    (no_trades["time_to_expiration"] > pd.Timedelta(days=daysTilExpiration)) &
    (((no_trades["theo"] - no_trades["market_price"]).abs()) >= edge)
]

# YES for where I would have theorically placed my trade and quantity  
sum_yes = ((1 - yes_filtered["theo_edge"]) * yes_filtered['theo_quantity']).sum()

# NO for where I would have theorically placed my trade and quantity  
sum_no = (no_filtered["theo_edge"] * no_filtered['theo_quantity']).sum()

total_pnl = sum_no - sum_yes

print("Filtered YES pnl:", sum_yes, " count:", len(yes_filtered))
print("Filtered NO pnl:", sum_no, " count:", len(no_filtered))
print("Filtered Total pnl:", total_pnl)


print(
    yes_filtered[
        ["ticker",  "time_to_expiration", "market_price", "theo", "edge", "theo_edge", "quantity", 'theo_quantity']
    ].head(10)
)

print(
    no_filtered[
        ["ticker", "expiration", "time_to_expiration", "market_price", "theo", "edge", "theo_edge", "quantity", 'theo_quantity']
    ].head(10)
)



# ************************* test different edge levels **********************

edge_values = np.arange(0.0, 0.31, 0.02)  # 0 → 0.30 in steps of 0.02

results = []

for edge in edge_values:
    df["theo_edge"] = df["theo"] + edge

    yes_filtered = yes_trades[
        (yes_trades["time_to_expiration"] > pd.Timedelta(days=1)) &
        ((yes_trades["theo"] - yes_trades["market_price"]).abs() >= edge)
    ]

    no_filtered = no_trades[
        (no_trades["time_to_expiration"] > pd.Timedelta(days=1)) &
        (((no_trades["theo"] - no_trades["market_price"]).abs()) >= edge)
    ]

    # YOUR pnl logic (unchanged)
    sum_yes = ((1 - yes_filtered["theo_edge"]) * yes_filtered["theo_quantity"]).sum()
    sum_no = (no_filtered["theo_edge"] * no_filtered["theo_quantity"]).sum()

    total_pnl = sum_no - sum_yes

    results.append({
        "edge": edge,
        "total_pnl": total_pnl,
        "yes_count": len(yes_filtered),
        "no_count": len(no_filtered)
    })


results_df = pd.DataFrame(results)
print(results_df)


plt.plot(results_df["edge"], results_df["total_pnl"])
plt.xlabel("Edge")
plt.ylabel("Total PnL")
plt.title("PnL vs Edge Threshold")
plt.grid()
plt.show()