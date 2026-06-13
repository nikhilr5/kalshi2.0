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