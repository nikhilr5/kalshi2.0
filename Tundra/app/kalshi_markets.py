"""Kalshi weather-market client for the Tundra app.

Thin wrapper over the Aston KalshiAPI that returns the open temperature buckets
for a city's series, parsed to continuous bounds with live book prices.
"""
import sys
from pathlib import Path

_ASTON = Path(__file__).resolve().parents[2] / "Aston"
_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
for _p in (str(_ASTON), str(_ANALYSIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kalshi_api import KalshiAPI          # noqa: E402
import weather_lib as wl                  # noqa: E402


class KalshiMarkets:
    def __init__(self, api=None):
        self.api = api or KalshiAPI()

    def weather_buckets(self, series, event_day=None, with_book=True):
        """Open buckets for a series. event_day filters to one day (e.g. '26JUN22').
        Returns rows sorted by lower bound:
          {ticker, event_day, sub_title, lo, hi, yes_bid, yes_ask, yes_mid, spread}
        Prices are dollars; lo/hi are the continuous bounds from parse_bucket."""
        markets = self.api.get_markets(series_ticker=series, status="open")
        out = []
        for m in markets:
            ticker = m.get("ticker")
            if not ticker:
                continue
            day = wl.ticker_day(ticker)
            if event_day and day != event_day:
                continue
            sub = m.get("yes_sub_title") or m.get("subtitle")
            bk = wl.parse_bucket(sub)
            if not bk:
                continue
            _, lo, hi = bk
            row = dict(ticker=ticker, event_day=day, sub_title=sub, lo=lo, hi=hi,
                       yes_bid=None, yes_ask=None, yes_mid=None, spread=None)
            if with_book:
                try:
                    mid, spread, by, bn = wl.book_mid_spread(self.api.get_orderbook(ticker, depth=1))
                    row.update(yes_bid=by, yes_ask=(1 - bn) if bn is not None else None,
                               yes_mid=mid, spread=spread)
                except Exception as e:
                    print(f"[kalshi] orderbook {ticker} failed: {e}")
            out.append(row)
        out.sort(key=lambda r: r["lo"] if r["lo"] > -9e8 else -1e9)
        return out
