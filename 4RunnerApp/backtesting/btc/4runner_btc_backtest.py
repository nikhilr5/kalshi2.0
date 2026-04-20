# backtest_btc_surface.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("../data/backtest_btc_results.csv")

df["created_time"] = pd.to_datetime(df["created_time"])
df["expiration"] = pd.to_datetime(df["expiration"])
df["time_to_expiration"] = df["expiration"] - df["created_time"]

# config
quantity = 100

# settled only
df_settled = df.dropna(subset=["settle_val"])

yes_trades = df_settled[df_settled["settlement"] == "yes"]
no_trades = df_settled[df_settled["settlement"] == "no"]

print(f"Total settled: {len(df_settled)} | YES: {len(yes_trades)} | NO: {len(no_trades)}")

# parameter ranges
edge_values = np.arange(0.0, 0.31, 0.02)
dte_values = np.arange(0.0, 2.2, 0.2)

# compute pnl for every combination
pnl_grid = np.zeros((len(dte_values), len(edge_values)))

print(f"\n{'':>8}", end="")
for e in edge_values:
    print(f"{'e='+f'{e:.2f}':>10}", end="")
print()

for i, dte in enumerate(dte_values):
    print(f"d={dte:.1f}  ", end="")
    for j, edge in enumerate(edge_values):
        yes_filtered = yes_trades[
            (yes_trades["time_to_expiration"] > pd.Timedelta(days=dte)) &
            ((yes_trades["theo"] - yes_trades["market_price"]).abs() >= edge)
        ]

        no_filtered = no_trades[
            (no_trades["time_to_expiration"] > pd.Timedelta(days=dte)) &
            ((no_trades["theo"] - no_trades["market_price"]).abs() >= edge)
        ]

        yes_pnl = ((1 - (yes_filtered["theo"] + edge)) * np.minimum(yes_filtered["quantity"], quantity)).sum()
        no_pnl = ((no_filtered["theo"] + edge) * np.minimum(no_filtered["quantity"], quantity)).sum()
        total_pnl = no_pnl - yes_pnl

        pnl_grid[i, j] = total_pnl
        print(f"{'$'+f'{total_pnl:,.0f}':>10}", end="")
    print()

# create meshgrid for surface plot
E, D = np.meshgrid(edge_values, dte_values)

# 3D surface plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# color by pnl: green positive, red negative
colors = np.where(pnl_grid >= 0, '#22c55e', '#ef4444')
norm = plt.Normalize(pnl_grid.min(), pnl_grid.max())
cmap = plt.cm.RdYlGn
facecolors = cmap(norm(pnl_grid))

surf = ax.plot_surface(E, D, pnl_grid, facecolors=facecolors, alpha=0.8, edgecolor='gray', linewidth=0.3)

ax.set_xlabel("Edge Threshold", fontsize=11, labelpad=10)
ax.set_ylabel("Min DTE (days)", fontsize=11, labelpad=10)
ax.set_zlabel("Total PnL ($)", fontsize=11, labelpad=10)
ax.set_title(f"BTC Weekly Bracket | PnL Surface\nqty cap: {quantity} | vol: 60%", fontsize=13)

# add colorbar
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(pnl_grid)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label("PnL ($)", fontsize=10)

ax.view_init(elev=25, azim=225)

fig.tight_layout()
fig.savefig("./graphs/btc_pnl_surface.png", dpi=150)
plt.close(fig)

# also make a heatmap (easier to read)
fig2, ax2 = plt.subplots(figsize=(14, 8))

im = ax2.imshow(pnl_grid, aspect='auto', cmap='RdYlGn', origin='lower',
                extent=[edge_values[0], edge_values[-1], dte_values[0], dte_values[-1]])

# label each cell
for i, dte in enumerate(dte_values):
    for j, edge in enumerate(edge_values):
        val = pnl_grid[i, j]
        x = edge_values[0] + (edge_values[-1] - edge_values[0]) * (j / (len(edge_values) - 1))
        y = dte_values[0] + (dte_values[-1] - dte_values[0]) * (i / (len(dte_values) - 1))
        color = "black" if abs(val) < abs(pnl_grid).max() * 0.6 else "white"
        ax2.text(x, y, f"${val:,.0f}", ha="center", va="center", fontsize=6, fontweight="bold", color=color)

ax2.set_xlabel("Edge Threshold", fontsize=12)
ax2.set_ylabel("Min DTE (days)", fontsize=12)
ax2.set_title(f"BTC Weekly Bracket | PnL Heatmap\nqty cap: {quantity} | vol: 60%", fontsize=13)
ax2.set_xticks(edge_values)
ax2.set_xticklabels([f"{e:.2f}" for e in edge_values], fontsize=8)
ax2.set_yticks(dte_values)
ax2.set_yticklabels([f"{d:.1f}" for d in dte_values], fontsize=8)

cbar2 = fig2.colorbar(im, ax=ax2)
cbar2.set_label("PnL ($)", fontsize=10)
cbar2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

fig2.tight_layout()
fig2.savefig("./graphs/btc_pnl_heatmap.png", dpi=150)
plt.close(fig2)

print(f"\nSaved to ./graphs/btc_pnl_surface.png")
print(f"Saved to ./graphs/btc_pnl_heatmap.png")