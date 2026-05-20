"""Backfill `fills.kalshi_ts` from /portfolio/fills.

Match strategy:
  Our fills table has no trade_id/order_id columns, so we match each
  null-kalshi_ts row to the API fill by (ticker, count, price,
  ts within ±30s).  Per-ticker the (count, price, ts) triple is
  essentially unique for our 1-lot strategy; ambiguity is rare.

Run:    python3 backfill_fills_kalshi_ts.py [--include-today]

Skips today's DB by default to avoid write contention with the live
recorder.  Today is already getting kalshi_ts populated correctly.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'Aston'))
from utility import list_eligible_dbs  # noqa: E402
from kalshi_api import KalshiAPI  # noqa: E402

SERIES_PREFIX = 'KXETH15M'
CUTOFF_DAY    = '26MAY15'
MATCH_WINDOW_S = 30
PRICE_TOL      = 1e-6


def fetch_all_account_fills(api: KalshiAPI) -> pd.DataFrame:
    """Paginate /portfolio/fills.  Returns DataFrame with normalised cols."""
    print(f'[backfill] pulling /portfolio/fills paginated...')
    fills = []
    cursor = None
    pages = 0
    while True:
        params = {'limit': 200}
        if cursor:
            params['cursor'] = cursor
        data = api._get('/portfolio/fills', params)
        page = data.get('fills', [])
        fills.extend(page)
        cursor = data.get('cursor')
        pages += 1
        if not cursor or not page:
            break
    df = pd.DataFrame(fills)
    if df.empty:
        return df
    df['ticker']     = df.get('market_ticker', df.get('ticker'))
    df['price']      = df['yes_price_dollars'].astype(float)
    df['count']      = df['count_fp'].astype(float)
    df['kalshi_ts']  = df['created_time']  # already ISO 8601
    df['ts_unix']    = df['ts'].astype(int)
    print(f'[backfill]   {len(df):,} account fills across {pages} pages')
    return df[['ticker', 'price', 'count', 'kalshi_ts', 'ts_unix',
                'trade_id', 'order_id']]


def backfill_db(path: Path, api_fills: pd.DataFrame) -> tuple[int, int]:
    """Apply backfill UPDATE to one per-day DB.  Returns (updated, ambiguous)."""
    conn = sqlite3.connect(str(path), timeout=30)
    cur = conn.cursor()
    # S3-cached older DBs predate the kalshi_ts migration that recorder.py
    # applies on open.  Add the column if missing (no-op otherwise).
    try:
        cur.execute("ALTER TABLE fills ADD COLUMN kalshi_ts TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    rows = cur.execute(
        "SELECT id, ts, ticker, count, price FROM fills "
        "WHERE kalshi_ts IS NULL OR kalshi_ts = ''"
    ).fetchall()
    if not rows:
        conn.close()
        return 0, 0

    local = pd.DataFrame(
        rows, columns=['id', 'ts', 'ticker', 'count', 'price'])
    local['ts_dt'] = pd.to_datetime(
        local['ts'], utc=True, format='ISO8601')

    api_by_ticker = {t: g.copy() for t, g in api_fills.groupby('ticker')}

    updated, ambiguous = 0, 0
    updates = []  # (kalshi_ts, id)
    for _, row in local.iterrows():
        g = api_by_ticker.get(row['ticker'])
        if g is None or g.empty:
            continue
        # Filter to matching count + price (with tolerance)
        cand = g[(abs(g['count'] - row['count']) < 1e-6)
                 & (abs(g['price'] - row['price']) < PRICE_TOL + 1e-6)]
        if cand.empty:
            continue
        # Pick the candidate closest in time within MATCH_WINDOW_S
        ts_unix = int(row['ts_dt'].timestamp())
        cand = cand.copy()
        cand['dt'] = (cand['ts_unix'] - ts_unix).abs()
        cand = cand[cand['dt'] <= MATCH_WINDOW_S]
        if cand.empty:
            continue
        if len(cand) > 1:
            ambiguous += 1
            # Still take the closest match; flag for awareness only.
        best = cand.nsmallest(1, 'dt').iloc[0]
        updates.append((best['kalshi_ts'], int(row['id'])))

    if updates:
        cur.executemany(
            "UPDATE fills SET kalshi_ts = ? WHERE id = ?",
            updates)
        conn.commit()
        updated = len(updates)
    conn.close()
    return updated, ambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-today', action='store_true',
                    help='also backfill today\'s DB (risks write conflict)')
    args = ap.parse_args()

    api = KalshiAPI()
    api_fills = fetch_all_account_fills(api)
    if api_fills.empty:
        print('[backfill] no API fills returned — abort')
        return

    # Filter to our series.
    api_fills = api_fills[api_fills['ticker'].str.startswith(SERIES_PREFIX)]
    print(f'[backfill]   {len(api_fills):,} fills match {SERIES_PREFIX}')

    files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY)
    today_suffix = datetime.now(timezone.utc).strftime('%y%b%d').upper()
    today_name = f'{SERIES_PREFIX}-{today_suffix}.db'

    total_updated, total_ambig = 0, 0
    for path in files:
        if path.name == today_name and not args.include_today:
            print(f'   {path.name}  (skipped — live recorder)')
            continue
        upd, amb = backfill_db(path, api_fills)
        total_updated += upd
        total_ambig += amb
        print(f'   {path.name}  updated={upd}  ambiguous={amb}')

    print(f'[backfill] done.  total_updated={total_updated}  '
          f'ambiguous_matches={total_ambig}')


if __name__ == '__main__':
    main()
