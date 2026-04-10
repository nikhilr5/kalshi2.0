import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = "1d307966d8b144479056777168799b58"

all_data = []
start = datetime(2026, 1, 1)
end = datetime(2026, 4, 7)

# 5 days per chunk = ~1400 bars, well under 5000 limit
chunk_days = 5
current = start

while current < end:
    chunk_end = min(current + timedelta(days=chunk_days), end)

    params = {
        "symbol": "XAU/USD",
        "interval": "5min",
        "start_date": current.strftime("%Y-%m-%d"),
        "end_date": chunk_end.strftime("%Y-%m-%d"),
        "outputsize": 5000,
        "apikey": API_KEY,
    }

    resp = requests.get("https://api.twelvedata.com/time_series", params=params)
    data = resp.json()

    if "values" in data:
        all_data.extend(data["values"])
        print(f"{current.date()} to {chunk_end.date()} | {len(data['values'])} bars | total: {len(all_data)}")
    else:
        print(f"{current.date()} to {chunk_end.date()} | Error: {data.get('message', data)}")

    current = chunk_end + timedelta(days=1)
    # rate limit - free tier is 8 requests/min
    time.sleep(8)

if all_data:
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    df.to_csv("gold_spot_5min.csv", index=False)
    print(f"\nTotal bars: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
else:
    print("No data collected")