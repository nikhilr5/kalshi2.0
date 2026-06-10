"""V2 orphan-rate diagnostic. Compares 6/9 (v2) vs 6/8 (v1) on the same metrics."""
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data")
DBS = {
    "v1_6_08": DATA / "KXETH15M-26JUN08.db",
    "v2_6_09": DATA / "KXETH15M-26JUN09.db",
}
ATT_V2 = DATA / "order_attempts-26JUN09.jsonl"


def parse_ts(s):
    return datetime.fromisoformat(s).timestamp()


def analyze_db(label, db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    t0, t1 = cur.execute("SELECT MIN(ts), MAX(ts) FROM order_events").fetchone()
    hours = (parse_ts(t1) - parse_ts(t0)) / 3600 if t0 else 0

    # All placed events with action
    placed = cur.execute("""
        SELECT ts, order_id, ticker, action, price, client_order_id
          FROM order_events WHERE event_type='placed' ORDER BY ts
    """).fetchall()

    # Did each order_id ever get a cancel or fill?
    cancelled_oids = set(r[0] for r in cur.execute(
        "SELECT DISTINCT order_id FROM order_events WHERE event_type='cancelled'"))
    filled_oids = set(r[0] for r in cur.execute(
        "SELECT DISTINCT order_id FROM order_events WHERE event_type='filled'"))

    # For each placed order, find earliest cancel ts and earliest fill ts (if any)
    next_evt = defaultdict(lambda: {"cancel": None, "fill": None})
    for ts, oid, etype in cur.execute(
        "SELECT ts, order_id, event_type FROM order_events "
        "WHERE event_type IN ('cancelled','filled') ORDER BY ts"
    ):
        k = "cancel" if etype == "cancelled" else "fill"
        if next_evt[oid][k] is None:
            next_evt[oid][k] = ts

    # Orphan definition (v1-baseline-compatible):
    # A placed order is "stale-no-cancel" if it had a fill arrive >30s (or >60s)
    # after place with NO cancel event recorded between place and fill.
    stale30, stale60, total_filled_placed = 0, 0, 0
    side_stale60 = Counter()
    side_total = Counter()
    for p_ts, oid, tkr, act, pr, coid in placed:
        side_total[act] += 1
        ne = next_evt.get(oid, {"cancel": None, "fill": None})
        f_ts = ne["fill"]
        if f_ts is None:
            continue
        total_filled_placed += 1
        rest = parse_ts(f_ts) - parse_ts(p_ts)
        c_ts = ne["cancel"]
        cancel_before_fill = c_ts is not None and parse_ts(c_ts) < parse_ts(f_ts)
        if rest > 30 and not cancel_before_fill:
            stale30 += 1
            if rest > 60:
                stale60 += 1
                side_stale60[act] += 1

    # Truly-lost: a placed event whose order_id NEVER got any followup
    # (no cancel, no fill — sits forever until market closes)
    truly_lost = 0
    truly_lost_by_side = Counter()
    for p_ts, oid, tkr, act, pr, coid in placed:
        if oid not in cancelled_oids and oid not in filled_oids:
            # exclude orders placed in the last 90s of the run (might still be active)
            if parse_ts(t1) - parse_ts(p_ts) > 90:
                truly_lost += 1
                truly_lost_by_side[act] += 1

    # Orphan-duplicate (LOST_ORDER_ID class A): two 'placed' within 2s
    # same (ticker, action, price) where the FIRST has no cancel
    placed_by_key = {}
    orphan_dup = 0
    orphan_dup_by_side = Counter()
    for p_ts, oid, tkr, act, pr, coid in placed:
        key = (tkr, act, round(pr, 4))
        prev = placed_by_key.get(key)
        if prev:
            prev_ts, prev_oid = prev
            if 0 < parse_ts(p_ts) - parse_ts(prev_ts) < 2.0:
                if prev_oid not in cancelled_oids:
                    orphan_dup += 1
                    orphan_dup_by_side[act] += 1
        placed_by_key[key] = (p_ts, oid)

    print(f"\n{'='*72}\n{label}: {db.name}\n{'='*72}")
    print(f"  window:           {hours:.2f}h  ({t0} → {t1})")
    print(f"  placed:           {len(placed):>6d}  "
          f"(buy={side_total.get('buy',0)}  sell={side_total.get('sell',0)})")
    print(f"  filled-after-place: {total_filled_placed:>6d}  "
          f"({100*total_filled_placed/max(1,len(placed)):.1f}%)")
    print(f"  stale-no-cancel >30s rest: {stale30:>5d}  "
          f"({stale30/max(1e-9,hours)*24:.0f}/day-equiv)")
    print(f"  stale-no-cancel >60s rest: {stale60:>5d}  "
          f"({stale60/max(1e-9,hours)*24:.0f}/day-equiv)")
    print(f"    by side @ >60s: buy={side_stale60.get('buy',0)} "
          f"sell={side_stale60.get('sell',0)}")
    if side_total.get('buy', 0) and side_total.get('sell', 0):
        buy_rate = side_stale60.get('buy', 0) / side_total['buy']
        sell_rate = side_stale60.get('sell', 0) / side_total['sell']
        print(f"    %-of-side:      buy={100*buy_rate:.2f}%  sell={100*sell_rate:.2f}%")
    print(f"  truly-lost (no cancel, no fill, placed >90s before EOF):"
          f" {truly_lost:>5d}  ({truly_lost/max(1e-9,hours)*24:.0f}/day-equiv)")
    print(f"    by side: buy={truly_lost_by_side.get('buy',0)} "
          f"sell={truly_lost_by_side.get('sell',0)}")
    print(f"  orphan-duplicate (class A, 2 placed <2s same px, "
          f"first uncancelled): {orphan_dup}")
    print(f"    by side: buy={orphan_dup_by_side.get('buy',0)} "
          f"sell={orphan_dup_by_side.get('sell',0)}")
    con.close()
    return {
        "hours": hours, "placed": len(placed), "stale30": stale30,
        "stale60": stale60, "truly_lost": truly_lost,
        "orphan_dup": orphan_dup,
    }


def attempt_summary():
    print(f"\n{'='*72}\norder_attempts-26JUN09.jsonl (V2 only)\n{'='*72}")
    total, success, fail, http_429 = 0, 0, 0, 0
    by_req = Counter()
    fail_by_err = Counter()
    success_orphan_candidates = 0  # success=1 but no companion 'placed' event
    by_ts_hour = Counter()
    with open(ATT_V2) as f:
        for ln in f:
            r = json.loads(ln)
            total += 1
            by_req[r["request_type"]] += 1
            hr = r["ts_request"][:13]
            by_ts_hour[hr] += 1
            if r["http_status"] == 429:
                http_429 += 1
            if r["success"] == 1:
                success += 1
            else:
                fail += 1
                fail_by_err[(r.get("error_code") or r.get("error_msg") or "?")[:40]] += 1
    print(f"  total attempts:   {total}")
    print(f"  by request_type:  {dict(by_req)}")
    print(f"  success: {success}  fail: {fail}  http_429: {http_429}")
    print(f"  fail breakdown (top 8):")
    for k, v in fail_by_err.most_common(8):
        print(f"    {v:>5d}  {k}")
    return total, fail, http_429


def success_path_bug_check():
    """V1 had: http=200, success=1, but no order_events 'placed' written.
    Cross-check today's success-create attempts vs order_events.placed."""
    con = sqlite3.connect(DBS["v2_6_09"])
    placed_coids = set(r[0] for r in con.execute(
        "SELECT DISTINCT client_order_id FROM order_events "
        "WHERE event_type='placed' AND client_order_id IS NOT NULL"))
    success_create_attempts = 0
    missing = 0
    missing_examples = []
    with open(ATT_V2) as f:
        for ln in f:
            r = json.loads(ln)
            if r["request_type"] != "create":
                continue
            if r["success"] != 1:
                continue
            coid = r.get("client_order_id")
            if not coid:
                continue
            success_create_attempts += 1
            if coid not in placed_coids:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append((coid, r["ts_request"], r["price"], r["action"]))
    print(f"\n{'='*72}\nSUCCESS-PATH BUG CHECK (v1 dominant orphan source)\n{'='*72}")
    print(f"  successful create attempts:           {success_create_attempts}")
    print(f"  coids missing from order_events.placed: {missing}  "
          f"({100*missing/max(1,success_create_attempts):.2f}%)")
    if missing_examples:
        print("  examples:")
        for coid, ts, pr, act in missing_examples:
            print(f"    {coid[:30]}  {ts}  {act}@{pr}")
    con.close()
    return missing, success_create_attempts


if __name__ == "__main__":
    r1 = analyze_db("V1 baseline (6/8)", DBS["v1_6_08"])
    r2 = analyze_db("V2 candidate (6/9)", DBS["v2_6_09"])
    attempt_summary()
    success_path_bug_check()

    print(f"\n{'='*72}\nNORMALIZED COMPARISON (per 24h-equivalent)\n{'='*72}")
    for metric in ["stale30", "stale60", "truly_lost", "orphan_dup"]:
        a = r1[metric] / max(1e-9, r1["hours"]) * 24
        b = r2[metric] / max(1e-9, r2["hours"]) * 24
        delta_pct = 100 * (b - a) / max(1e-9, a)
        print(f"  {metric:>14s}: v1={a:6.1f}/day  v2={b:6.1f}/day  "
              f"Δ={delta_pct:+.1f}%")
