"""Live test: are batch creates billed at 10 or 100 tokens per item?
Bucket capacity = 600 (Advanced). A 7-order batch is impossible at
100/item (700 > 600), trivial at 10/item (70). Orders: buy yes @ $0.01,
count=1, post_only, on an open KXBTC15M market (Aston doesn't trade BTC).
Cleans up with a batch cancel, which also probes cancel pricing (7x20=140).
"""
import sys, time, uuid, json
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/Aston")
from kalshi_api import KalshiAPI

api = KalshiAPI()

# 1. Find an open BTC 15m market
markets = api.get_markets(series_ticker="KXBTC15M", status="open")
if not markets:
    sys.exit("no open KXBTC15M market found")
ticker = markets[0]["ticker"]
print(f"[test] using {ticker}")

# 2. Let the shared bucket refill (Aston is live; 3s of idle from US ~ 900
#    tokens of refill headroom, capped at 600)
print("[test] waiting 4s for bucket refill...")
time.sleep(4)

# 3. Fire one batch of 7 creates
orders = [{
    "ticker": ticker,
    "side": "yes",
    "action": "buy",
    "yes_price_dollars": "0.010",
    "count": 1,
    "client_order_id": f"btest_{uuid.uuid4()}",
    "type": "limit",
    "time_in_force": "good_till_canceled",
    "post_only": True,
} for _ in range(7)]

t0 = time.perf_counter()
resp = api._post("/portfolio/orders/batched", {"orders": orders})
dt = (time.perf_counter() - t0) * 1000
print(f"[test] batch create returned in {dt:.0f}ms")
print(json.dumps(resp, indent=2)[:3000])

# 4. Tally results + collect order_ids for cleanup
created_ids = []
status = resp.get("status_code")
items = resp.get("orders") or []
ok = 0
for it in items:
    order = (it or {}).get("order") or it or {}
    oid = order.get("order_id")
    if oid:
        created_ids.append(oid)
        ok += 1
print(f"\n[test] HTTP {status}; per-item successes: {ok}/7")

# 5. Cleanup — batch cancel whatever was created
if created_ids:
    time.sleep(1)
    c = api.cancel_orders_batched(created_ids)
    cok = sum(1 for it in (c.get("orders") or []) if not (it or {}).get("error"))
    print(f"[test] batch cancel: {cok}/{len(created_ids)} cancelled "
          f"(HTTP {c.get('status_code')})")
    print(json.dumps(c, indent=2)[:1500])

# 6. Verdict
print("\n=== VERDICT ===")
if ok == 7:
    print("All 7 created in one batch -> >600 tokens is impossible,")
    print("so batch creates are NOT 100/item. Consistent with 10/item.")
elif 0 < ok < 7:
    print(f"Partial success ({ok}/7) -> consistent with ~100/item")
    print("(bucket could only fund that many). Check per-item errors above.")
else:
    print("Batch rejected outright -> inspect status/errors above.")
