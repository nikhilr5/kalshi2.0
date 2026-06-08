"""Root-cause analysis: stale fills where OSM never issued a cancel.

Classifies each adverse-stale fill (rest>=5s AND adverse theo drift>1¢)
into one of three failure modes:

  A. Orphan-duplicate (LOST_ORDER_ID): two 'placed' events appear
     close in time for the same (ticker, action, price); the FIRST is
     never cancelled.  Signature of `_send_place` retrying after a
     local error (Errno 35 / 'invalid_order' / etc.) where the request
     actually landed on Kalshi but OSM didn't record the server_order_id.

  B. Tolerance gate (PRICE_SPACE_GATE): osm._reconcile_action returns
     'keep' because |raw_desired - resting_price| < tolerance (1¢) at
     the time of fill — even though raw theo drifted >1¢.

  C. BBO clamp (CLAMP_PIN): desired = strategy2 BBO-clamped value which
     hadn't moved during the resting period.

Reproducible: writes findings.txt next to this script.
"""
import math
import sqlite3
from pathlib import Path

DB = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data/KXETH15M-26MAY22.db")
EDGE_BID, EDGE_ASK, TOLERANCE = 0.07, 0.05, 0.01


def parse_ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def round_to_tick(price, side):
    grid = 1000.0 if (price < 0.10 or price > 0.90) else 100.0
    if side == "bid":
        return max(math.floor(price * grid) / grid, 0.001)
    return min(math.ceil(price * grid) / grid, 0.999)


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # ------------------------------------------------------------------
    # 1. Build orphan set: 'placed' pair within 2s same (ticker,action,price)
    #    where the FIRST never has a 'cancelled' event.
    # ------------------------------------------------------------------
    rows = cur.execute("""
        SELECT ts, order_id, ticker, action, price, client_order_id
          FROM order_events
         WHERE event_type='placed' ORDER BY ts
    """).fetchall()
    placed_by_key = {}
    orphan_coids = {}  # coid -> twin coid
    for ts, oid, ticker, action, price, coid in rows:
        key = (ticker, action, round(price, 4))
        prev = placed_by_key.get(key)
        if prev:
            prev_ts, prev_oid, prev_coid = prev
            if 0 < parse_ts(ts) - parse_ts(prev_ts) < 2.0:
                cn = cur.execute("""
                    SELECT COUNT(*) FROM order_events
                     WHERE order_id=? AND event_type='cancelled'
                """, (prev_oid,)).fetchone()[0]
                if cn == 0:
                    orphan_coids[prev_coid] = coid
        placed_by_key[key] = (ts, oid, coid)
    print(f"Orphan client_order_ids (no cancel ever, has duplicate twin): {len(orphan_coids)}")

    # ------------------------------------------------------------------
    # 2. Stale fills with adverse drift
    # ------------------------------------------------------------------
    fills = cur.execute("""
        SELECT f.ts, f.ticker, f.action, f.price, f.client_order_id,
               oe_p.ts, oe_p.order_id, oe_p.price
          FROM fills f
          JOIN order_events oe_p
            ON oe_p.client_order_id = f.client_order_id
           AND oe_p.event_type='placed'
         ORDER BY f.ts
    """).fetchall()
    cases = []
    for f_ts, tkr, act, fp, coid, p_ts, oid, pp in fills:
        rest = parse_ts(f_ts) - parse_ts(p_ts)
        if rest < 5.0:
            continue

        # theo at place + fill
        theo = cur.execute("""
            SELECT theo FROM theo_state WHERE ticker=? AND ts BETWEEN ? AND ? ORDER BY ts
        """, (tkr, p_ts, f_ts)).fetchall()
        if not theo:
            continue
        theo_p, theo_f = theo[0][0], theo[-1][0]
        if act == "buy":
            drift = theo_p - theo_f
            side = "bid"
            des_pre = theo_f - EDGE_BID
        else:
            drift = theo_f - theo_p
            side = "ask"
            des_pre = theo_f + EDGE_ASK
        if drift <= 0.01:
            continue

        bbo_f = cur.execute("""
            SELECT yes_bid, yes_ask FROM kalshi_book
             WHERE ticker=? AND ts <= ? ORDER BY ts DESC LIMIT 1
        """, (tkr, f_ts)).fetchone() or (0.0, 0.0)
        bbo_p = cur.execute("""
            SELECT yes_bid, yes_ask FROM kalshi_book
             WHERE ticker=? AND ts <= ? ORDER BY ts DESC LIMIT 1
        """, (tkr, p_ts)).fetchone() or (0.0, 0.0)

        if side == "bid":
            raw = min(des_pre, bbo_f[0])
        else:
            raw = max(des_pre, bbo_f[1])
        snapped = round_to_tick(raw, side)

        gate_skips = abs(raw - pp) < TOLERANCE

        # Classify
        if coid in orphan_coids:
            mode = "A_ORPHAN"
        elif gate_skips:
            mode = "B_TOLERANCE_GATE"
        elif ((side == "bid" and abs(raw - bbo_f[0]) < 1e-9 and bbo_f[0] == bbo_p[0])
              or (side == "ask" and abs(raw - bbo_f[1]) < 1e-9 and bbo_f[1] == bbo_p[1])):
            mode = "C_BBO_CLAMP"
        else:
            mode = "?_OTHER"

        cases.append({
            "mode": mode, "f_ts": f_ts, "p_ts": p_ts, "rest": rest,
            "side": side, "drift": drift,
            "pp": pp, "raw": raw, "snapped": snapped,
            "theo_p": theo_p, "theo_f": theo_f,
            "bbo_p": bbo_p, "bbo_f": bbo_f,
            "oid": oid, "coid": coid,
            "twin_coid": orphan_coids.get(coid),
        })

    by_mode = {}
    for c in cases:
        by_mode.setdefault(c["mode"], []).append(c)

    print()
    print("=" * 95)
    print(f"ADVERSE-STALE FILLS (rest>=5s, |theo drift|>1¢):  TOTAL = {len(cases)}")
    print("=" * 95)
    for mode in ["A_ORPHAN", "B_TOLERANCE_GATE", "C_BBO_CLAMP", "?_OTHER"]:
        n = len(by_mode.get(mode, []))
        print(f"  {mode}: {n}  ({100*n/max(1,len(cases)):.1f}%)")

    # ------------------------------------------------------------------
    # Examples for each mode
    # ------------------------------------------------------------------
    for mode in ["A_ORPHAN", "B_TOLERANCE_GATE", "C_BBO_CLAMP", "?_OTHER"]:
        bucket = by_mode.get(mode, [])
        if not bucket:
            continue
        bucket.sort(key=lambda c: -c["drift"])
        print()
        print("-" * 95)
        print(f"{mode}: 5 worst-drift examples")
        print("-" * 95)
        for c in bucket[:5]:
            print(f"  side={c['side']} rest={c['rest']:.0f}s drift={100*c['drift']:.1f}¢ "
                  f"placed@{c['pp']*100:.1f}¢  raw_des={c['raw']*100:.1f}¢  "
                  f"|diff|={abs(c['raw']-c['pp'])*100:.1f}¢  "
                  f"theo {c['theo_p']*100:.1f}→{c['theo_f']*100:.1f}¢")
            print(f"    oid={c['oid'][:8]}  coid={c['coid'][:24]}  twin={(c['twin_coid'] or '-')[:24]}")

    # ------------------------------------------------------------------
    # Cross-check: orphans against order_attempts (only post-17:15 has data)
    # ------------------------------------------------------------------
    orphan_in_window = [c for c in by_mode.get("A_ORPHAN", [])
                        if c["p_ts"] > "2026-05-22T17:15:00"]
    print()
    print(f"Orphan cases in order_attempts window (post-17:15): {len(orphan_in_window)}")
    for c in orphan_in_window:
        att = cur.execute("""
            SELECT ts_request, success, error_msg, server_order_id
              FROM order_attempts WHERE client_order_id=?
        """, (c["coid"],)).fetchall()
        print(f"  coid={c['coid'][:24]} attempts:")
        for ts_r, ok, err, sid in att:
            print(f"    success={ok}  err={(err or '')[:60]}  server_oid={sid}")

    con.close()


if __name__ == "__main__":
    main()
