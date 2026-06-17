import sys
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import load_fills, load_orders, load_book, load_theo, day_range, fetch_settlements_from_api
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI


'''
Overview:
    I want to evaluate a bunch of features (signals) on fills I've had over the past month to determine which have a significant impact on the outcome of the event.
Model(s):
    Logistic Regression
Question:
    Which features have a significant impact on a order/fill predicting the outcome.
Features:
    - B0 = quote age
    - B1 = moneyness at fill time
    - B2 = spread (ask-bid) at fill time
    - B3 = book imbalance at fill time
    - B4 = hour of day at fill time
    - B5 = (theo - mid) when placing
    - B6 = seconds til expiry
Null Hypthoesis:
    Each feature has no impact on the outcome.
'''
START = '2026-05-16'


def build_day(day):
    """One day -> one row per fill with all features. None if no usable data.
    Loading per-day keeps the multi-GB book/theo tables out of memory."""
    fills = load_fills(day).sort_values("ts")
    book  = load_book(day).sort_values("ts")
    if fills.empty or book.empty:
        return None
    orders = load_orders(day).sort_values("ts")
    theos  = load_theo(day).sort_values("ts")

    fills = fills[['ts', 'ticker', 'client_order_id', 'kalshi_ts']]

    # mid right before the order was placed
    orders = orders[orders['event_type'] == 'placed'][['ts', 'ticker', 'client_order_id', 'kalshi_ts']]
    orders = pd.merge_asof(orders, book[['ts', 'ticker', 'mid']], on='ts', by='ticker', direction='backward')

    theos = theos[theos['seconds_to_expiry'] > 0][['ts', 'ticker', 'theo', 'seconds_to_expiry']]

    # last book + theo before each fill, then the order that produced the fill
    fo = pd.merge_asof(fills, book[['ts', 'ticker', 'yes_bid', 'yes_ask', 'bid_size', 'ask_size']],
                       on='ts', by='ticker', direction='backward')
    fo = pd.merge_asof(fo, theos, on='ts', by='ticker', direction='backward')
    fo = pd.merge(fo, orders, on=['client_order_id', 'ticker'], suffixes=['_f', '_o'])
    if fo.empty:
        return None

    # B0 quote age
    fo['quote_age_sec'] = (pd.to_datetime(fo['kalshi_ts_f'], utc=True, format='ISO8601')
                           - pd.to_datetime(fo['kalshi_ts_o'], utc=True, format='ISO8601')).dt.total_seconds()
    fo = fo[fo['quote_age_sec'] > 0]
    # B1 moneyness = continuous mid at fill
    fo['moneyness'] = (fo['yes_bid'] + fo['yes_ask']) / 2
    # B2 spread
    fo['spread'] = fo['yes_ask'] - fo['yes_bid']
    fo = fo[fo['spread'] > 0]
    # B3 book imbalance
    denom = fo['bid_size'] + fo['ask_size']
    fo['imbalance'] = (fo['bid_size'] - fo['ask_size']) / denom.where(denom > 0)
    # B4 hour of day (CT)
    fo['ts_f'] = pd.to_datetime(fo['ts_f'], utc=True)
    fo['hour'] = fo['ts_f'].dt.tz_convert("America/Chicago").dt.hour
    fo['day'] = fo['ts_f'].dt.date
    # B5 theo - mid at placement
    fo['theo_mid_diff'] = fo['theo'] - fo['mid']

    return fo[['ticker', 'day', 'quote_age_sec', 'moneyness', 'spread',
               'imbalance', 'hour', 'theo_mid_diff', 'seconds_to_expiry']]


df = pd.concat([f for f in (build_day(d) for d in day_range(START, 'today')) if f is not None],
               ignore_index=True)

# outcome = 1 if the market settled YES
cache = Path(__file__).resolve().parent / ".settlements_cache.json"
settlements = fetch_settlements_from_api(df['ticker'].unique().tolist(), KalshiAPI(), cache_path=cache)
df['outcome'] = df['ticker'].map(settlements)

df = df.dropna().copy()
df['outcome'] = df['outcome'].astype(int)

model = smf.logit(
    "outcome ~ quote_age_sec + moneyness + spread + imbalance "
    "+ C(hour) + theo_mid_diff + seconds_to_expiry",
    data=df,
).fit(cov_type="cluster", cov_kwds={"groups": df["day"]})

print(model.summary())
