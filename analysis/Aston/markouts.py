import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import (load_all_data, calculate_markouts)
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

markout_secs = [1, 10, 30, 60, 120, 180]
result = calculate_markouts(fills, book, markout_secs)

print('1s Markout:', result['markout_1s'].mean(), '\n10s Markout:', result['markout_10s'].mean(),
    '\n30s Markout', result['markout_30s'].mean(), 
    '\n120s Markout', result['markout_120s'].mean(), 
    '\n180s Markout', result['markout_180s'].mean())


