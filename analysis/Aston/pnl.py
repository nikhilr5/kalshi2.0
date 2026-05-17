import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import (load_all_data, calculate_markouts, plot_markout_heatmaps)
import pandas as pd
import numpy as np
from utility import fetch_settlements_from_api

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"   # inclusive lower bound, YYMONDD


# =============================================================================
# Load + clean
# =============================================================================
theo, book, spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)

#get settlements
api = KalshiAPI()
unique_tickers = list(fills['ticker'].unique())
settlements = fetch_settlements_from_api(
    unique_tickers, api,
    cache_path=Path(__file__).resolve().parent / ".settlements_cache.json"
)

#get per day pnl
fills['cash_flow'] = np.where(fills['action'] == 'buy',
                              -fills['count'] * fills['price'],
                              +fills['count'] * fills['price'])
fills['signed_count'] = np.where(fills['action'] == 'buy',
                                  fills['count'], -fills['count'])

per_market = fills.groupby('ticker').agg(
    cash_flow=('cash_flow', 'sum'),
    net_position=('signed_count', 'sum'),
)
per_market['settlement'] = per_market.index.map(settlements)
per_market['pnl'] = per_market['cash_flow'] + per_market['net_position'] * per_market['settlement']

per_market = per_market.sort_values('pnl', ascending=False).reset_index(drop=True)
print(per_market.head())

print(per_market.tail())