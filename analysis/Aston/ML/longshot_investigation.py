'''
Overview:
    Want to investigate if there is edge in fading the longshot yes trades
Idea: 
    Retail traders are willing to trade for a longshot bet.
H0:
    The true probability is equal to the market ask in the [0.1 - 0.35] region.
'''

import pandas as pd
import sys
from pathlib import Path
import numpy as np
import datetime as dt
import statsmodels.formula.api as smf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import load_theo, load_book, fetch_settlements_from_api, load_fills, day_range
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

def close_ts(tk):
      return dt.datetime.strptime(tk.split('-')[1], '%y%b%d%H%M').replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=4)

date = '2026-5-16'
frames = []
for day in day_range(date, 'today'):
    book = load_book(day)
    if book.empty:
        continue
    book['ttc'] = (book['ticker'].map(close_ts) - book['ts']).dt.total_seconds()
    snap = book[book['ttc'] >= 300].sort_values('ts').groupby('ticker', as_index=False).tail(1)  # 1 row/market at ~5min out
    snap = snap[(snap['yes_ask'] > 0.1) & (snap['yes_ask'] < 0.35)]                                # cheap band
    frames.append(snap)

book = pd.concat(frames, ignore_index=True)

#add outcomes
cache = Path(__file__).resolve().parent.parent / ".settlements_cache.json"
settlements = fetch_settlements_from_api(book['ticker'].unique().tolist(),
                                         KalshiAPI(), cache_path=cache)
book['outcome'] = book['ticker'].map(settlements)

book['logit_yes_ask'] = np.log(book['yes_ask'])
book['day'] = pd.to_datetime(book['ts'], utc=True).dt.date


book = book[['outcome', 'logit_yes_ask', 'day']]
book = book.dropna().copy()

model = smf.logit("outcome ~ logit_yes_ask", data=book).fit(cov_type="cluster", cov_kwds={"groups": book["day"]})
print(model.summary())
