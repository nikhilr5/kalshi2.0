#pull each trade and see where and what I traded
#see if I guessed correctly. Is that buy or sell bucket positive?


import sqlite3, pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import (list_eligible_dbs, fetch_settlements_from_api)

import sys; sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/Aston")
from kalshi_api import KalshiAPI


frames = []
for p in list_eligible_dbs("KXETH15M", "26MAY15"):
    conn = sqlite3.connect(str(p))
    try:
        frames.append(pd.read_sql("SELECT * FROM fills", conn))
    except Exception:
        pass            # MAY30/31 empty, JUN09 corrupt
    finally:
        conn.close()
fills = pd.concat(frames, ignore_index=True)
fills["ts"] = pd.to_datetime(fills["ts"], utc=True, format="ISO8601")
fills = fills[fills["ts"] >= "2026-05-15"]

print("Fill Count", len(fills))


tickers = fills["ticker"].unique().tolist()
s = fetch_settlements_from_api(tickers, KalshiAPI(),
        cache_path="/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/.settlements_cache.json")
fills["outcome"] = fills["ticker"].map(s)
fills = fills[fills["outcome"].notna()]

fills['correct'] = np.where((fills['action'] == 'buy'), (fills['outcome'] == 1), (fills['outcome'] == 0))
fills['correct'] = np.where(fills['correct'] == 1, 1, 0)
fills = fills[['correct', 'price', 'action']]
fills = fills[fills['price'] >= .1]
fills = fills[fills['price'] <= .9]
fills_grouped_by = fills.groupby(by=['price', 'action']).agg(['mean', 'count'])



price  = fills_grouped_by.index.get_level_values('price').values
action = fills_grouped_by.index.get_level_values('action').values
correct = fills_grouped_by[('correct', 'mean')].values

fills_grouped_by['edge'] = np.where(
    action == 'buy',
    correct - price,
    correct - (1 - price))


fills_grouped_by = fills_grouped_by.sort_values('edge', ascending=False)

with pd.option_context('display.max_rows', None):                                                                       
      print(fills_grouped_by)

print(fills_grouped_by[('correct', 'count')].sum())


# ---- roll the per-0.01 rows up into 0.1-wide buckets (count-weighted) ----
roll = fills_grouped_by.copy()
roll.columns = ['correct', 'count', 'edge']        # flatten the MultiIndex cols
roll = roll.reset_index()                           # price, action out of index
roll['px_bin'] = pd.cut(roll['price'], np.arange(0.1, 0.91, 0.1))

buckets = (roll.groupby(['px_bin', 'action'], observed=True)
               .apply(lambda g: pd.Series({
                   'correct': np.average(g['correct'], weights=g['count']),
                   'edge':    np.average(g['edge'],    weights=g['count']),
                   'count':   g['count'].sum(),
               }), include_groups=False))

with pd.option_context('display.max_rows', None):
    print(buckets)


# ---- scatter: edge vs price bucket, buy/sell split, dot size ~ count ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plot_df = buckets.reset_index()
plot_df['xlab'] = plot_df['px_bin'].astype(str)
xlabels = [str(b) for b in plot_df['px_bin'].cat.categories]
xpos = {lab: i for i, lab in enumerate(xlabels)}

fig, ax = plt.subplots(figsize=(11, 6))
for action, color in [('buy', '#ef4444'), ('sell', '#22c55e')]:
    sub = plot_df[plot_df['action'] == action]
    ax.scatter([xpos[l] for l in sub['xlab']], sub['edge'],
               s=sub['count'] / 8, c=color, alpha=0.65,
               edgecolors='k', linewidths=0.5, label=action)
ax.axhline(0, color='#888', lw=1, ls='--')
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels, rotation=30, ha='right')
ax.set_xlabel('price bucket')
ax.set_ylabel('edge ($/contract, maker, gross=net)')
ax.set_title('Edge by price bucket — buy vs sell  (dot size ∝ fill count)')
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
out = Path(__file__).resolve().parent / "AgentGenerated" / "settlement_buckets_edge.png"
fig.savefig(out, dpi=120)
print(f"[saved] {out}")