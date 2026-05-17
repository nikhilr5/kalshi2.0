import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import (load_all_data, calculate_markouts, plot_markout_heatmaps)
import pandas as pd
import numpy as np

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"   # inclusive lower bound, YYMONDD


# =============================================================================
# Load + clean
# =============================================================================
theo, book, spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)


fills = fills[['ts', 'ticker', 'count', 'action', 'price']]
fills = fills.sort_values('ts').reset_index(drop=True)

book = book.sort_values('ts').reset_index(drop=True)

markout_secs = [1, 10, 30, 60, 120, 180, 300]

#get seconds til expirations
result = pd.merge_asof(
    fills.sort_values('ts'),
    theo[['ts', 'ticker', 'seconds_to_expiry']].sort_values('ts'),
    on='ts',
    by='ticker',
    direction='backward',
    suffixes=['_r', '_t']
)
result = calculate_markouts(result, book, markout_secs)

#print markout results
print('1s Markout:', result['markout_1s'].mean(), '\n10s Markout:', result['markout_10s'].mean(),
    '\n30s Markout', result['markout_30s'].mean(), 
    '\n120s Markout', result['markout_120s'].mean(), 
    '\n180s Markout', result['markout_180s'].mean(),
    '\n300s Markout', result['markout_300s'].mean())


#bucket by minutes til expiration
result = result.dropna(subset=['seconds_to_expiry'])
result['minutes_til_expiration'] = (result['seconds_to_expiry'] / 60).astype(int) + 1

#plot as heat map and pass in results
plot_markout_heatmaps(result, markout_secs)
