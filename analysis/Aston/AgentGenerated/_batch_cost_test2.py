"""Discriminator: two back-to-back batches of 5 creates.
10/item  -> 50+50 = 100 tokens: BOTH succeed (even with Aston noise).
100/item -> 500+500: first succeeds, second MUST 429 (bucket 600).
"""
import sys, time, uuid, json
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/Aston")
from kalshi_api import KalshiAPI

api = KalshiAPI()

markets = api.get_markets(series_ticker="KXBTC15M", status="open")
ticker = markets[0]["ticker"]
print(f"[test] using {ticker}")
print("[test] waiting 15s (client backoff 10s + bucket refill)...")
time.sleep(15)

def fire_batch(n, label):
    orders = [{
        "ticker": ticker, "side": "yes", "action": "buy",
        "yes_price_dollars": "0.010", "count": 1,
        "client_order_id": f"btest_{uuid.uuid4()}",
        "type": "limit", "time_in_force": "good_till_canceled",
        "post_only": True,
    } for _ in range(n)]
    resp = api._post("/portfolio/orders/batched", {"orders": orders})
    status = resp.get("status_code")
    ids = []
    for it in (resp.get("orders") or []):
        order = (it or {}).get("order") or it or {}
        oid = order.get("order_id")
        if oid:
            ids.append(oid)
    print(f"[{label}] HTTP {status}, created {len(ids)}/{n}")
    if status != 201:
        print(f"   body: {json.dumps(resp)[:300]}")
    return status, ids

s1, ids1 = fire_batch(5, "batch1")
s2, ids2 = fire_batch(5, "batch2")   # immediately after

all_ids = ids1 + ids2
if all_ids:
    time.sleep(2)
    c = api.cancel_orders_batched(all_ids)
    print(f"[cleanup] batch cancel HTTP {c.get('status_code')}, "
          f"{len(all_ids)} ids submitted")

print("\n=== VERDICT ===")
if s1 == 201 and s2 == 201:
    print("Both 5-order batches succeeded back-to-back (>=1000 tokens at "
          "100/item is impossible) -> batch creates ARE cheap (~10/item).")
elif s1 == 201 and s2 == 429:
    print("First batch ok, second 429 -> batch creates cost ~100/item. "
          "No batching discount.")
else:
    print(f"s1={s1}, s2={s2} -> inconclusive (likely Aston drained bucket); rerun.")
