"""15-min up/down market discovery for Kalshi crypto series.

The KXBTC15M / KXETH15M / KXSOL15M / KXXRP15M series each list one
single-strike "up/down" market per 15-minute window.  Strike is the
target price set at window open (not in the ticker — pulled from the
market's `floor_strike` field).

Each market is its own event (one market per event), so we use the
flat /markets endpoint with the series_ticker filter and sort by
close_time ascending — the soonest-to-close open market is the
currently-active trading window.
"""

from datetime import datetime, timezone
from kalshi_api import KalshiAPI


# Crypto series available in the 15-min family.  `coinbase_product` is
# the Coinbase websocket symbol used to feed spot.
SERIES_15M = [
    {"ticker": "KXBTC15M", "name": "Bitcoin",  "coinbase_product": "BTC-USD"},
    {"ticker": "KXETH15M", "name": "Ethereum", "coinbase_product": "ETH-USD"},
    {"ticker": "KXSOL15M", "name": "Solana",   "coinbase_product": "SOL-USD"},
    {"ticker": "KXXRP15M", "name": "XRP",      "coinbase_product": "XRP-USD"},
]


def discover_open_markets(api: KalshiAPI, series_ticker: str) -> list:
    """Pull every open market for a 15-min series, sorted by close_time asc.

    Returns a list of dicts (the raw market objects from Kalshi's API).
    Each carries `ticker`, `close_time`, `floor_strike`, `strike_type`,
    `yes_bid`, `yes_ask`, etc.  Pagination handled by `api.get_markets`.
    """
    markets = api.get_markets(series_ticker=series_ticker, status="open")
    return sorted(markets, key=lambda m: m.get("close_time", ""))


def get_active_market(api: KalshiAPI, series_ticker: str) -> dict | None:
    """Return the current 15-min market — soonest-to-close open one.

    Filters out anything that already closed (clock skew safety).  None
    if no open market exists (rare gap between cycles or API hiccup).
    """
    now = datetime.now(tz=timezone.utc)
    for m in discover_open_markets(api, series_ticker):
        close_str = m.get("close_time", "")
        if not close_str:
            continue
        try:
            close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if close_utc > now:
            return m
    return None


def parse_strike(market: dict) -> float:
    """Extract the threshold price from a 15-min up/down market.

    Kalshi's API may surface the strike under different keys depending
    on the product spec.  Try them in order and fall back to 0.0 if
    none match.  `cap_strike` is the same as `floor_strike` on these
    single-threshold contracts (yes = price above threshold).
    """
    for key in ("floor_strike", "cap_strike", "strike"):
        val = market.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 0.0


def seconds_to_close(market: dict) -> float:
    """Seconds remaining until this market settles. <= 0 if past close."""
    close_str = market.get("close_time", "")
    if not close_str:
        return 0.0
    try:
        close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    return (close_utc - datetime.now(tz=timezone.utc)).total_seconds()
