import json
import datetime as dt
from collections import Counter, defaultdict

F = "/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data/order_attempts-26JUN10.jsonl"

def p(s):
    return dt.datetime.fromisoformat(s)

rows = [json.loads(l) for l in open(F)]
for r in rows:
    r["t"] = p(r["ts_request"])
rows.sort(key=lambda r: r["t"])

t0, t1 = rows[0]["t"], rows[-1]["t"]

# --- find era boundary: first create with count==3 ---
first_c3 = next((r["t"] for r in rows if r["request_type"] == "create" and r["count"] == 3), None)

def era(r):
    # single file: only v2 here. Split at first 3-lot create.
    if first_c3 and r["t"] >= first_c3:
        return "v2_3lot_cap30"
    return "v2_1lot"

ERAS = ["v2_1lot", "v2_3lot_cap30"]

print(f"span {t0} -> {t1}  ({(t1-t0).total_seconds()/60:.1f} min), {len(rows)} attempts")
print(f"first count=3 create at {first_c3}\n")

# ---- 1. 429 rate per era ----
print("=== 1. 429 rate by era ===")
for e in ERAS:
    er = [r for r in rows if era(r) == e]
    if not er:
        print(f"{e}: (no rows)"); continue
    creates = [r for r in er if r["request_type"] == "create"]
    cancels = [r for r in er if r["request_type"] == "cancel"]
    c429 = [r for r in creates if r["http_status"] == 429]
    x429 = [r for r in cancels if r["http_status"] == 429]
    dur_min = (er[-1]["t"] - er[0]["t"]).total_seconds() / 60 or 1
    print(f"{e}: {len(er)} attempts over {dur_min:.1f}min | "
          f"creates={len(creates)} cancels={len(cancels)} | "
          f"create-429={len(c429)} ({100*len(c429)/max(len(creates),1):.1f}% of creates) "
          f"cancel-429={len(x429)}")
    print(f"    creates/min={len(creates)/dur_min:.1f}  cancels/min={len(cancels)/dur_min:.1f}  "
          f"429/min={(len(c429)+len(x429))/dur_min:.1f}")

# ---- 2. write-token rate, 1s windows ----
print("\n=== 2. write-token rate (10*create+2*cancel) per 1s window, by era ===")
def pctile(xs, q):
    if not xs: return 0
    xs = sorted(xs); i = min(len(xs)-1, int(q*len(xs)))
    return xs[i]
for e in ERAS:
    er = [r for r in rows if era(r) == e]
    if not er: continue
    buckets = defaultdict(int)
    for r in er:
        sec = int(r["t"].timestamp())
        buckets[sec] += 10 if r["request_type"] == "create" else 2
    vals = list(buckets.values())
    print(f"{e}: p50={pctile(vals,.5)} p95={pctile(vals,.95)} p99={pctile(vals,.99)} "
          f"max={max(vals)}  (windows>100tok: {sum(1 for v in vals if v>100)}/{len(vals)})")

# ---- 3. 10s-sawtooth signature around 429 ----
print("\n=== 3. burst structure around 429 episodes (sawtooth test) ===")
all_creates = sorted([r for r in rows if r["request_type"] == "create"], key=lambda r: r["t"])
c429 = [r for r in all_creates if r["http_status"] == 429]
print(f"total create-429: {len(c429)}")
# cluster 429s into episodes (gap > 3s starts new episode)
episodes = []
for r in c429:
    if not episodes or (r["t"] - episodes[-1][-1]["t"]).total_seconds() > 3:
        episodes.append([r])
    else:
        episodes[-1].append(r)
print(f"429 episodes (>3s gap splits): {len(episodes)}")
# inter-episode gaps — sawtooth would cluster near ~10s
gaps = []
for i in range(1, len(episodes)):
    g = (episodes[i][0]["t"] - episodes[i-1][0]["t"]).total_seconds()
    gaps.append(g)
near10 = [g for g in gaps if 8 <= g <= 13]
print(f"inter-episode start gaps: n={len(gaps)} median={pctile(gaps,.5):.1f}s "
      f"in[8,13]s={len(near10)} ({100*len(near10)/max(len(gaps),1):.0f}%)")

# ---- 4. create:cancel ratio + per-side ----
print("\n=== 4. create:cancel ratio + per-side split by era ===")
for e in ERAS:
    er = [r for r in rows if era(r) == e]
    if not er: continue
    cr = sum(1 for r in er if r["request_type"] == "create")
    ca = sum(1 for r in er if r["request_type"] == "cancel")
    # side: creates have action buy(bid)/sell(ask); cancels often have null ticker/action
    buys = sum(1 for r in er if r["request_type"]=="create" and r["action"]=="buy")
    sells = sum(1 for r in er if r["request_type"]=="create" and r["action"]=="sell")
    print(f"{e}: create:cancel = {cr}:{ca} ({cr/max(ca,1):.2f})  create buy(bid)={buys} sell(ask)={sells}")

# ---- 5. same-price repeat-attempt within 1s after a failure ----
print("\n=== 5. repeat (ticker,action,price) create within 1s of a failed create ===")
for e in ERAS:
    er = sorted([r for r in rows if era(r)==e and r["request_type"]=="create"], key=lambda r: r["t"])
    repeats = 0
    last_fail = {}  # (ticker,action,price) -> t of last failure
    for r in er:
        key = (r["ticker"], r["action"], r["price"])
        if key in last_fail and (r["t"] - last_fail[key]).total_seconds() <= 1.0:
            repeats += 1
        if r["success"] != 1:
            last_fail[key] = r["t"]
    print(f"{e}: same-(ticker,action,price) create-after-fail within 1s = {repeats} "
          f"of {len(er)} creates ({100*repeats/max(len(er),1):.1f}%)")

# ---- bonus: status code breakdown per era ----
print("\n=== status code breakdown by era (creates) ===")
for e in ERAS:
    cc = Counter(r["http_status"] for r in rows if era(r)==e and r["request_type"]=="create")
    print(f"{e}: {dict(cc)}")
