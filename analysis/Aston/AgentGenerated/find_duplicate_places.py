"""Find OSM duplicate-place orphans:
  - Two 'placed' events for same (ticker, action, price) within 2s
  - First one NEVER receives a 'cancelled' event
  - First eventually FILLED → stale orphan

Indicates OSM lost track of the order_id (overwritten in resting_*).
"""
import sqlite3
from pathlib import Path

DB = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data/KXETH15M-26MAY22.db")


def parse_ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Pull all 'placed' events; iterate in Python to find pairs.
    rows = cur.execute("""
        SELECT ts, order_id, ticker, action, price, client_order_id
          FROM order_events
         WHERE event_type='placed'
         ORDER BY ts
    """).fetchall()
    print(f"Total placed events: {len(rows)}")

    # Group sequential pairs by (ticker, action, price); find pairs
    # within 2s and the first is never cancelled.
    placed_by_key = {}   # most recent (ticker, action, price) -> (ts, oid, coid)
    pairs = []
    for ts, oid, ticker, action, price, coid in rows:
        key = (ticker, action, round(price, 4))
        prev = placed_by_key.get(key)
        if prev:
            prev_ts, prev_oid, prev_coid = prev
            dt = parse_ts(ts) - parse_ts(prev_ts)
            if 0 < dt < 2.0:
                pairs.append({
                    "ticker": ticker, "action": action, "price": price,
                    "ts1": prev_ts, "oid1": prev_oid, "coid1": prev_coid,
                    "ts2": ts,      "oid2": oid,      "coid2": coid,
                    "dt_ms": dt * 1000.0,
                })
        placed_by_key[key] = (ts, oid, coid)

    print(f"Same-price duplicate-place pairs within 2s: {len(pairs)}")

    orphan_pairs = []
    for p in pairs:
        cancelled = cur.execute("""
            SELECT COUNT(*) FROM order_events
             WHERE order_id=? AND event_type='cancelled'
        """, (p["oid1"],)).fetchone()[0]
        filled = cur.execute("""
            SELECT COUNT(*) FROM order_events
             WHERE order_id=? AND event_type='filled'
        """, (p["oid1"],)).fetchone()[0]
        if cancelled == 0 and filled > 0:
            ft = cur.execute("""
                SELECT ts FROM fills WHERE client_order_id=?
            """, (p["coid1"],)).fetchone()
            if ft:
                rest_s = parse_ts(ft[0]) - parse_ts(p["ts1"])
                orphan_pairs.append({**p, "rest_s": rest_s, "fill_ts": ft[0]})

    print(f"Orphan pairs (first never cancelled, eventually filled): {len(orphan_pairs)}")
    stale = [p for p in orphan_pairs if p["rest_s"] >= 5.0]
    print(f"  ...stale (rest>=5s): {len(stale)}")
    print()

    for i, p in enumerate(stale[:15]):
        print(f"--- #{i+1} {p['ticker'][-25:]} {p['action']} @ {p['price']*100:.1f}¢ "
              f"  dt={p['dt_ms']:.1f}ms  rest={p['rest_s']:.0f}s ---")
        print(f"  ORPHAN  oid={p['oid1'][:8]} placed {p['ts1'][:23]} → filled {p['fill_ts'][:23]}")
        print(f"  TRACKED oid={p['oid2'][:8]} placed {p['ts2'][:23]}")

    con.close()


if __name__ == "__main__":
    main()
