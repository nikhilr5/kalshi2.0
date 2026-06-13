import pandas as pd
import sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import load_theo, load_book, fetch_settlements_from_api, load_fills
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

'''
Null Hypthoesis: 
    There is no difference between the brier score of the theo to the brier score of the mid.
Definition: 
    Brier Score: Is (outcome - predicted ) ^2.
    A good brier score is 0 since its calculating the error in the prediction.
Process:
    Calculate the brier score for theo and mid. For each tick take the difference of the two calculated field.
    Group by each ticker (market) and take the mean of the calculated difference. 
    Check if that mean is significant by calculating a F-value (how many std devs from 0) and getting the p-value (area under the curve) from it.
'''

#load data past this date
date = '2026-06-12'
theo = load_theo(date, until='today')
book = load_book(date, until='today')

# drop post-close (sec<=0, theo=NaN) and any pre-open rows
theo = theo[(theo['seconds_to_expiry'] > 0) & (theo['seconds_to_expiry'] <= 900)]
theo = theo[['ts', 'ticker', 'theo', 'seconds_to_expiry', 'spot', 'strike', 'sigma']].sort_values('ts')
book = book[['ts', 'ticker', 'mid']].sort_values('ts')

# for each theo row, the most recent book mid at-or-before its ts, same ticker
theo_book = pd.merge_asof(theo, book, on='ts', by='ticker', direction='backward')

# settlement is ONE value per ticker (not time-varying) -> map on ticker, not asof
cache = Path(__file__).resolve().parent / ".settlements_cache.json"
settlements = fetch_settlements_from_api(theo_book['ticker'].unique().tolist(),
                                         KalshiAPI(), cache_path=cache)
theo_book['outcome'] = theo_book['ticker'].map(settlements)

theo_book['brier_theo'] = (theo_book['outcome'] - theo_book['theo']) ** 2
theo_book['brier_mid'] = (theo_book['outcome'] - theo_book['mid']) ** 2
theo_book['d'] =  theo_book['brier_mid'] - theo_book['brier_theo']

# moneyness: |z| = |log(S/K)| / (sigma * sqrt(T_years)).  positive d = theo better.
T = theo_book['seconds_to_expiry'] / (365.25 * 24 * 3600)
theo_book['abs_z'] = (np.log(theo_book['spot'] / theo_book['strike']).abs()
                      / (theo_book['sigma'] * np.sqrt(T)))
theo_book['zbin'] = pd.cut(theo_book['abs_z'], [0, 0.25, 0.5, 1.0, 2.0, np.inf],
                           labels=['ATM<0.25', '0.25-0.5', '0.5-1.0', '1.0-2.0', '>2.0'])

def f_test(g):
    n = len(g)
    mean = g['d'].mean()
    SSR_mean = np.sum(g['d'] ** 2)
    SSR_fit = np.sum((g['d'] - mean) ** 2)
    F = (SSR_mean - SSR_fit) / (SSR_fit / (n - 1))
    p = stats.f.sf(F, 1, n - 1)
    return n, mean, F, p

per_ticker = theo_book.groupby('ticker').agg(d=('d', 'mean'), zbin=('zbin', 'first'))
print('Since:', date, '  (positive mean = theo better)')
print('  ALL       ', '%5d  mean=%+.5f  F=%7.3f  p=%.5g' % f_test(per_ticker))
for zb in ['ATM<0.25', '0.25-0.5', '0.5-1.0', '1.0-2.0', '>2.0']:
    sub = per_ticker[per_ticker['zbin'] == zb]
    if len(sub) > 1:
        print('  %-9s ' % zb, '%5d  mean=%+.5f  F=%7.3f  p=%.5g' % f_test(sub))
# the regime you actually quote in
print('  excl|z|>2 ', '%5d  mean=%+.5f  F=%7.3f  p=%.5g'
      % f_test(per_ticker[per_ticker['zbin'] != '>2.0']))