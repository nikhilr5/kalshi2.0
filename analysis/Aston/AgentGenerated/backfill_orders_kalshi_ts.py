"""Backfill `order_events.kalshi_ts` from /portfolio/orders.

Match strategy: exact by `order_id` (no ambiguity).  Per row:
  placed     → kalshi_ts = order.created_time
  cancelled  → kalshi_ts = order.last_update_time (if status='canceled')
  filled     → kalshi_ts = order.last_update_time (if status='executed')

Run:    python3 backfill_orders_kalshi_ts.py [--include-today]

Skips today's DB by default (live recorder writer).  ~10 min wall time
expected on 6-day window.
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'Aston'))
from utility import list_eligible_dbs, parse_day_suffix  # noqa: E402
from kalshi_api import KalshiAPI  # noqa: E402

SERIES_PREFIX = 'KXETH15M'
CUTOFF_DAY    = '26MAY15'


def fetch_all_account_orders(api: KalshiAPI, min_ts: int) -> pd.DataFrame:
    """Paginate /portfolio/orders without status filter (= all)."""
    print(f'[backfill] pulling /portfolio/orders min_ts={min_ts} ...')
    orders = []
    cursor = None
    pages = 0
    t0 = time.time()
    while True:
        params = {'limit': 200, 'min_ts': min_ts}
        if cursor:
            params['cursor'] = cursor
        data = api._get('/portfolio/orders', params)
        page = data.get('orders', [])
        orders.extend(page)
        cursor = data.get('cursor')
        pages += 1
        if pages % 50 == 0:
            elapsed = time.time() - t0
            print(f'   page {pages}  n={len(orders):,}  '
                  f'elapsed={elapsed:.0f}s')
        if not cursor or not page:
            break
        # Light throttle to be polite to the live trading API.
        time.sleep(0.05)
    df = pd.DataFrame(orders)
    print(f'[backfill]   {len(df):,} account orders across {pages} pages '
          f'({time.time() - t0:.0f}s)')
    return df


def backfill_db(path: Path, api_orders: pd.DataFrame) -> dict:
    """Apply backfill UPDATE to one per-day DB.  Returns counts per event_type."""
    conn = sqlite3.connect(str(path), timeout=30)
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE order_events ADD COLUMN kalshi_ts TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    rows = cur.execute(
        "SELECT id, order_id, event_type FROM order_events "
        "WHERE (kalshi_ts IS NULL OR kalshi_ts = '') "
        "  AND order_id IS NOT NULL AND order_id != ''"
    ).fetchall()
    if not rows:
        conn.close()
        return {'total': 0}

    local = pd.DataFrame(rows, columns=['id', 'order_id', 'event_type'])
    by_oid = {o['order_id']: o for o in api_orders.to_dict('records')}

    updates = []
    counts = {'placed': 0, 'cancelled': 0, 'filled': 0, 'no_api_match': 0,
              'status_mismatch': 0}
    for _, row in local.iterrows():
        o = by_oid.get(row['order_id'])
        if o is None:
            counts['no_api_match'] += 1
            continue
        evt = row['event_type']
        if evt == 'placed':
            ts = o.get('created_time')
        elif evt == 'cancelled':
            ts = (o.get('last_update_time')
                  if (o.get('status') or '').lower() in ('canceled', 'cancelled')
                  else None)
            if ts is None:
                counts['status_mismatch'] += 1
                continue
        elif evt == 'filled':
            ts = (o.get('last_update_time')
                  if (o.get('status') or '').lower() == 'executed'
                  else None)
            if ts is None:
                counts['status_mismatch'] += 1
                continue
        else:
            continue
        if ts:
            updates.append((ts, int(row['id'])))
            counts[evt] = counts.get(evt, 0) + 1

    if updates:
        cur.executemany(
            "UPDATE order_events SET kalshi_ts = ? WHERE id = ?",
            updates)
        conn.commit()
    conn.close()
    counts['total'] = len(updates)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-today', action='store_true',
                    help='also backfill today\'s DB (risks write conflict)')
    args = ap.parse_args()

    cutoff_date = parse_day_suffix(CUTOFF_DAY)
    min_ts = int(datetime(cutoff_date.year, cutoff_date.month,
                          cutoff_date.day, tzinfo=timezone.utc).timestamp())

    api = KalshiAPI()
    api_orders = fetch_all_account_orders(api, min_ts)
    if api_orders.empty:
        print('[backfill] no API orders returned — abort')
        return

    api_orders = api_orders[
        api_orders['ticker'].str.startswith(SERIES_PREFIX)]
    print(f'[backfill]   {len(api_orders):,} orders match {SERIES_PREFIX}')

    files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY)
    today_suffix = datetime.now(timezone.utc).strftime('%y%b%d').upper()
    today_name = f'{SERIES_PREFIX}-{today_suffix}.db'

    grand_total = 0
    for path in files:
        if path.name == today_name and not args.include_today:
            print(f'   {path.name}  (skipped — live recorder)')
            continue
        c = backfill_db(path, api_orders)
        grand_total += c.get('total', 0)
        print(f'   {path.name}  placed={c.get("placed",0)}  '
              f'cancelled={c.get("cancelled",0)}  '
              f'filled={c.get("filled",0)}  '
              f'no_match={c.get("no_api_match",0)}  '
              f'status_mismatch={c.get("status_mismatch",0)}  '
              f'total={c.get("total",0)}')

    print(f'[backfill] done.  total_updated={grand_total}')


if __name__ == '__main__':
    main()
