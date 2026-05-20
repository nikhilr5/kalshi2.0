import sqlite3
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import fetch_settlements_from_api

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

path = '../backtesting/data/KXETH15M-26MAY20.db'
conn = sqlite3.connect(str(path))

fills = pd.read_sql(f"SELECT * FROM fills;",conn)
orders= pd.read_sql(f"SELECT * FROM order_events;", conn)
theo = pd.read_sql(f"SELECT * FROM theo_state;",conn)

theo['forecasted_vol'] = (
    0.0314
    + 0.4485 * theo['rv_15m']
    + 0.1293 * theo['rv_30m']
    + 0.1843 * theo['rv_4h']
    + 0.1149 * theo['rv_24h']
)
theo = theo[theo['seconds_to_expiry'] > 0]


#left join orders placed on canceled
order_and_cancelled = pd.merge(orders[orders['event_type'] == 'placed'], 
    orders[orders['event_type'] == 'cancelled'],
    how='left',
    on='client_order_id',
    suffixes=['_o', '_c'])

#orders that were actually filled
filled_orders = order_and_cancelled[order_and_cancelled['event_type_c'] != 'cancelled']

#join on fills
fills_and_orders = pd.merge(
    filled_orders,
    fills,
    how='inner',
    on='client_order_id',
    suffixes=['_o', '_f'])

print(theo.columns)
fills_and_orders = fills_and_orders[['client_order_id', 'kalshi_ts_o', 'kalshi_ts', 'price', 'strike', 'ticker', 'action']]

fills_and_orders['theo_when_placed'] = np.where(
    fills_and_orders['action'] == 'buy',
    fills_and_orders['price'] + 0.07,
    fills_and_orders['price'] - 0.05
)

fills_and_orders['kalshi_ts_o'] =  pd.to_datetime(fills_and_orders['kalshi_ts_o'])
fills_and_orders['kalshi_ts'] =  pd.to_datetime(fills_and_orders['kalshi_ts'])
theo['ts'] = pd.to_datetime(theo['ts'], utc=True, format='ISO8601')
fills_and_orders['quote_ms'] = (fills_and_orders['kalshi_ts'] - fills_and_orders['kalshi_ts_o']).dt.total_seconds() * 1000

print(fills_and_orders.columns)

#check if there are any theos where the theo moves against the theo used to placed by more than 1c
results = []
for _, row in fills_and_orders.iterrows():
    mask = (theo['ts'] >= row['kalshi_ts_o']) & (theo['ts'] <= row['kalshi_ts'])
    matches = theo[mask].copy()
    matches['source_id'] = row.name   # or another key
    results.append(matches)
results = pd.concat(results, ignore_index=True)
results = results.merge(
    fills_and_orders[['action', 'price', 'theo_when_placed', 'quote_ms', 'client_order_id', 'kalshi_ts_o', 'kalshi_ts']],
    left_on='source_id', right_index=True,
)

diff = np.where(
    results['action'] == 'buy',
    results['theo_when_placed'] - results['theo'],
    results['theo'] - results['theo_when_placed'],
)
results['cancel_trigger'] = (diff > 0.01)
results['canceled_amount'] = diff

should_have_cancelled = results[results['cancel_trigger'] == True]

#largest amount the theo moved against our theo when we placed
idx = should_have_cancelled.groupby('client_order_id')['canceled_amount'].idxmax()
magnitude = should_have_cancelled.loc[idx]

magnitude = magnitude[['ts',  'kalshi_ts_o', 'kalshi_ts', 'client_order_id', 'theo', 'theo_when_placed', 'canceled_amount']]
magnitude['theo_to_fill_ms'] = (magnitude['kalshi_ts'] - magnitude['ts']).dt.total_seconds() * 1000

magnitude = magnitude.sort_values('theo_to_fill_ms', ascending=False)
print(magnitude['theo_to_fill_ms'].median())
print(len(magnitude))

