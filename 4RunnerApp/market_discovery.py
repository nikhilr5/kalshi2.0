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
                if brackets:
                    events.append({
                        "event_ticker": event_ticker,
                        "date": date,
                        "close_time": brackets[0].get("close_time", ""),
                        "markets": brackets,
                        "num_brackets": len(brackets),
                    })
        except Exception:
            continue

    return events


def parse_strike(ticker: str) -> float:
    for prefix in ["-B", "-T"]:
        if prefix in ticker:
            try:
                return float(ticker.split(prefix)[1])
            except ValueError:
                return 0.0
    return 0.0
