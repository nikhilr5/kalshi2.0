import requests
import time
import base64
import csv
from pathlib import Path
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
ACCESS_KEY = "2bc651e6-3882-4206-b539-93540910df06"
PRIVATE_KEY_PATH = Path.home() / "private_key.pem"

with open(PRIVATE_KEY_PATH, "rb") as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)

def sign(timestamp_ms, method, path):
    message = f"{timestamp_ms}{method}{path}"
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")

def auth_headers(method, path):
    ts = int(time.time() * 1000)
    return {
        "KALSHI-ACCESS-KEY": ACCESS_KEY,
        "KALSHI-ACCESS-SIGNATURE": sign(ts, method, path),
        "KALSHI-ACCESS-TIMESTAMP": str(ts),
    }

def fetch_all_trades(endpoint, params_base):
    all_trades = []
    cursor = None
    while True:
        path = f"/trade-api/v2/{endpoint}"
        headers = auth_headers("GET", path)
        params = {**params_base, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{API_BASE}/{endpoint}", headers=headers, params=params, timeout=15)
        data = resp.json()
        trades = data.get("trades", [])
        all_trades.extend(trades)
        cursor = data.get("cursor")
        if not cursor or not trades:
            break
        time.sleep(0.2)
    return all_trades

month_map = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
             7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}

# generate weekday event tickers from Jan 1 to today
start = datetime(2026, 1, 1)
end = datetime.now()
current = start

event_tickers = []
while current <= end:
    if current.weekday() < 5:  # Mon-Fri
        yy = str(current.year)[2:]
        mm = month_map[current.month]
        dd = f"{current.day:02d}"
        event_tickers.append(f"KXBTC-{yy}{mm}{dd}17")
    current += timedelta(days=1)

print(f"Checking {len(event_tickers)} potential events...")

# fetch markets for each event, only keep brackets
all_markets = []
found_events = 0

for i, event_ticker in enumerate(event_tickers):
    if i % 20 == 0:
        print(f"  {i}/{len(event_tickers)}... found {found_events} events, {len(all_markets)} bracket markets")

    path = "/trade-api/v2/markets"
    headers = auth_headers("GET", path)
    params = {"event_ticker": event_ticker, "limit": 200}

    try:
        resp = requests.get(f"{API_BASE}/markets", headers=headers, params=params, timeout=10)
        data = resp.json()
        markets = data.get("markets", [])
        if markets:
            found_events += 1
            # only keep bracket markets
            brackets = [m for m in markets if "-B" in m["ticker"]]
            # verify it's weekly by checking open_time vs close_time
            for m in brackets:
                open_time = m.get("open_time", "")
                close_time = m.get("close_time", "")
                if open_time and close_time:
                    try:
                        ot = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                        ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                        duration_days = (ct - ot).total_seconds() / 86400
                        m["duration_days"] = round(duration_days, 1)
                    except:
                        m["duration_days"] = 0
                else:
                    m["duration_days"] = 0

            # only keep weekly (5-8 days duration)
            weekly = [m for m in brackets if 4 <= m.get("duration_days", 0) <= 9]
            all_markets.extend(weekly)
    except:
        pass

    time.sleep(0.1)

print(f"\nFound {found_events} events")
print(f"Weekly bracket markets: {len(all_markets)}")

# filter to markets with volume
with_vol = [m for m in all_markets if float(m.get("volume_fp", "0")) > 0]
print(f"With volume: {len(with_vol)}")

# show duration breakdown
durations = set(m.get("duration_days", 0) for m in all_markets)
print(f"Durations seen: {sorted(durations)}")

# show sample
for m in all_markets[:5]:
    print(f"  {m['ticker']} | {m.get('yes_sub_title', '?')} | dur: {m.get('duration_days')}d | vol: {m.get('volume_fp', '0')}")

# build metadata
market_info = {}
for m in all_markets:
    market_info[m["ticker"]] = {
        "result": m.get("result", ""),
        "expiration": m.get("latest_expiration_time", ""),
        "status": m.get("status", ""),
        "yes_sub_title": m.get("yes_sub_title", ""),
        "expiration_value": m.get("expiration_value", ""),
        "open_time": m.get("open_time", ""),
        "close_time": m.get("close_time", ""),
        "duration_days": m.get("duration_days", 0),
    }

# fetch trades
print(f"\nFetching trades for {len(with_vol)} markets...")
all_trades = []

for i, m in enumerate(with_vol):
    ticker = m["ticker"]
    if i % 10 == 0:
        print(f"[{i+1}/{len(with_vol)}] {ticker}...")

    try:
        trades = fetch_all_trades("markets/trades", {"ticker": ticker})
    except Exception as e:
        print(f"  Error: {e}")
        trades = []

    info = market_info.get(ticker, {})
    for t in trades:
        t["settlement_result"] = info.get("result", "")
        t["expiration"] = info.get("expiration", "")
        t["market_status"] = info.get("status", "")
        t["yes_sub_title"] = info.get("yes_sub_title", "")
        t["expiration_value"] = info.get("expiration_value", "")
        t["duration_days"] = info.get("duration_days", 0)

    all_trades.extend(trades)
    if trades:
        print(f"  {len(trades)} trades")

print(f"\nTotal trades collected: {len(all_trades)}")

# write CSV
output_path = "data/kxbtc_weekly_bracket_trades.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "trade_id", "ticker", "strike", "created_time",
        "yes_price", "no_price", "quantity", "taker_side",
        "yes_sub_title", "expiration", "expiration_value",
        "settlement_result", "market_status", "duration_days"
    ])

    for t in all_trades:
        ticker = t["ticker"]
        parts = ticker.split("-B")
        strike = parts[1] if len(parts) >= 2 else ""

        writer.writerow([
            t.get("trade_id", ""),
            ticker,
            strike,
            t.get("created_time", ""),
            t.get("yes_price_dollars", ""),
            t.get("no_price_dollars", ""),
            t.get("count_fp", ""),
            t.get("taker_side", ""),
            t.get("yes_sub_title", ""),
            t.get("expiration", ""),
            t.get("expiration_value", ""),
            t.get("settlement_result", ""),
            t.get("market_status", ""),
            t.get("duration_days", 0),
        ])

print(f"Saved to {output_path}")

# summary
print("\nTrades by event:")
event_counts = {}
for t in all_trades:
    event = t["ticker"].split("-B")[0] if "-B" in t["ticker"] else t["ticker"]
    event_counts[event] = event_counts.get(event, 0) + 1

for ev in sorted(event_counts.keys()):
    print(f"  {ev}: {event_counts[ev]} trades")