import sqlite3
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import fetch_settlements_from_api

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

path = '../backtesting/data/KXETH15M-26MAY20.db'
conn = sqlite3.connect(str(path))
fills = pd.read_sql(f"SELECT * FROM fills WHERE kalshi_ts IS NOT NULL ORDER BY ts;",conn)
print(fills.head())

orders= pd.read_sql(f"SELECT * FROM order_events WHERE kalshi_ts IS NOT NULL ORDER BY ts;", conn)
print(orders.head())

print(orders.columns)
print(len(orders))

# only care about cancels
cancels = orders[orders['event_type'] == 'cancelled']

#orders that were filled but we tried to cancel them
too_slow_to_cancel = pd.merge(
    fills,
    cancels,
    on='client_order_id',
    how='inner',
    suffixes=['_fills', '_cancels'])

print(len(fills))
print(len(too_slow_to_cancel))

print(too_slow_to_cancel.columns)

too_slow_to_cancel['kalshi_ts_fills'] = pd.to_datetime(too_slow_to_cancel['kalshi_ts_fills'])
too_slow_to_cancel['kalshi_ts_cancels'] = pd.to_datetime(too_slow_to_cancel['kalshi_ts_cancels'])


too_slow_to_cancel['cancel_behind_ms'] = (too_slow_to_cancel['kalshi_ts_cancels']- too_slow_to_cancel['kalshi_ts_fills']).dt.total_seconds() * 1000
print(too_slow_to_cancel[['client_order_id', 'event_ticker', 'price_fills', 'cancel_behind_ms', 'kalshi_ts_fills', 'kalshi_ts_cancels']].head())

markets = list(too_slow_to_cancel['event_ticker'])

settlements = fetch_settlements_from_api(
    markets,
    KalshiAPI(),
    cache_path=Path(__file__).resolve().parent / ".settlements_cache.json",
)

print(settlements)