"""Pull public trade tape from Kalshi REST for every ticker in the
validation window and cache locally for counterfactual fill simulation.

One JSON per ticker under `_trades_cache/`.  Resumable: existing files
are skipped.  Re-run safely.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import load_all_data  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI  # noqa: E402


SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
CACHE_DIR     = (Path(__file__).resolve().parent.parent.parent
                 / "backtesting" / "_trades_cache")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Discover tickers from the recorder DBs in the window.
    theo, _book, _spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
    tickers = sorted(set(theo['ticker'].dropna().unique())
                     | set(fills['ticker'].dropna().unique())
                     | set(events['ticker'].dropna().unique()))
    print(f"[trades] {len(tickers)} unique tickers in window ≥ {CUTOFF_DAY}")

    todo = [t for t in tickers if not (CACHE_DIR / f"{t}.json").exists()]
    print(f"[trades] {len(todo)} need fetching ({len(tickers) - len(todo)} already cached)")
    if not todo:
        return

    api = KalshiAPI()
    t0 = time.time()
    total_trades = 0

    for i, ticker in enumerate(todo, 1):
        try:
            # The wrapper caps at limit=1000 by default; bump for safety.
            trades = api.get_trades(ticker, limit=10_000)
        except Exception as e:
            print(f"[trades] {ticker}: API error: {e}")
            continue

        out_path = CACHE_DIR / f"{ticker}.json"
        with out_path.open("w") as f:
            json.dump(trades, f)
        total_trades += len(trades)

        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(todo) - i) / rate if rate else 0
            print(f"[trades] {i}/{len(todo)}  "
                  f"last={ticker} n={len(trades)}  "
                  f"total_trades={total_trades:,}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s")

        # Light hand on the API while live trading is running.
        time.sleep(0.05)

    print(f"[trades] done. {total_trades:,} trades fetched across "
          f"{len(todo)} new tickers in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
