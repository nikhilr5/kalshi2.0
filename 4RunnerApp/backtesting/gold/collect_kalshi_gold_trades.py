# fix_settlements.py
# re-fetch correct settlement results from Kalshi API
import pandas as pd
import requests
import time
import base64
from pathlib import Path
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

# load trades
trades = pd.read_csv("data/kxgoldmon_trades.csv")
tickers = trades["ticker"].unique()
print(f"Fetching settlement data for {len(tickers)} contracts...")

# fetch each contract's details
results = {}
for i, ticker in enumerate(tickers):
    if i % 20 == 0:
        print(f"  {i}/{len(tickers)}...")

    path = f"/trade-api/v2/markets/{ticker}"
    headers = auth_headers("GET", path)

    try:
        resp = requests.get(f"{API_BASE}/markets/{ticker}", headers=headers, timeout=10)
        m = resp.json().get("market", {})
        results[ticker] = {
            "result": m.get("result", ""),
            "expiration_value": m.get("expiration_value", ""),
            "yes_sub_title": m.get("yes_sub_title", ""),
            "close_time": m.get("close_time", ""),
            "status": m.get("status", ""),
        }
    except Exception as e:
        print(f"  Error for {ticker}: {e}")
        results[ticker] = {"result": "", "expiration_value": "", "yes_sub_title": "", "close_time": "", "status": ""}

    time.sleep(0.15)

# save to CSV for reference
settlement_df = pd.DataFrame([
    {"ticker": t, **info} for t, info in results.items()
])
settlement_df.to_csv("data/settlements.csv", index=False)
print(f"\nSaved {len(settlement_df)} settlement records to data/settlements.csv")

# print some examples
print("\nSample settlements:")
for _, row in settlement_df.head(20).iterrows():
    print(f"  {row['ticker']} | result: {row['result']} | exp_val: {row['expiration_value']} | yes: {row['yes_sub_title']}")