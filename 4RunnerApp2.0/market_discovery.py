"""Discovers weekly bracket events and their markets on Kalshi."""

from datetime import datetime, timedelta
from kalshi_api import KalshiAPI

MONTH_MAP = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
             7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}


def discover_weekly_events(api: KalshiAPI, series: str = "KXBTC", weeks_ahead: int = 2) -> list:
    """Find active weekly bracket events by checking upcoming weekday dates."""
    today = datetime.now()
    events = []

    for day_offset in range(weeks_ahead * 7):
        date = today + timedelta(days=day_offset)
        if date.weekday() >= 5:
            continue

        yy = str(date.year)[2:]
        mm = MONTH_MAP[date.month]
        dd = f"{date.day:02d}"
        event_ticker = f"{series}-{yy}{mm}{dd}17"

        try:
            markets = api.get_markets_for_event(event_ticker)
            if markets:
                brackets = [m for m in markets if "-B" in m["ticker"]]
                tails = [m for m in markets if "-T" in m["ticker"]]
                if brackets:
                    events.append({
                        "event_ticker": event_ticker,
                        "date": date,
                        "close_time": brackets[0].get("close_time", ""),
                        "markets": brackets + tails,
                        "num_brackets": len(brackets),
                    })
        except Exception:
            continue

    return events


def discover_events_for_series(api: KalshiAPI, series_ticker: str) -> list:
    """Discover all active bracket/above-below events for a series via the markets API.

    Uses the /markets endpoint with series_ticker filter, which works for any
    series (KXBTC, KXBTCD, KXETH, etc.) without hardcoding date formats.
    """
    markets = api.get_markets(series_ticker=series_ticker, status="open")
    # Include bracket (-B) and tail/above-below (-T) markets
    relevant = [m for m in markets if "-B" in m["ticker"] or "-T" in m["ticker"]]

    # Group by event_ticker
    events_map: dict[str, dict] = {}
    for m in relevant:
        et = m.get("event_ticker", "")
        if not et:
            continue
        if et not in events_map:
            events_map[et] = {
                "event_ticker": et,
                "close_time": m.get("close_time", ""),
                "markets": [],
                "num_brackets": 0,
            }
        events_map[et]["markets"].append(m)

    for ev in events_map.values():
        ev["num_brackets"] = len(ev["markets"])

    return sorted(events_map.values(), key=lambda e: e.get("close_time", ""))


def parse_strike(ticker: str) -> float:
    """Extract strike from ticker. Returns the raw value (e.g. 83799.99)."""
    for prefix in ["-B", "-T"]:
        if prefix in ticker:
            try:
                return float(ticker.split(prefix)[1])
            except ValueError:
                return 0.0
    return 0.0


def display_strike(raw_strike: float) -> float:
    """Convert raw ticker strike to display strike.

    KXBTCD tickers use e.g. 83799.99 to mean "$83,800 or above".
    Round up to nearest dollar for display.
    """
    import math
    return math.ceil(raw_strike)
