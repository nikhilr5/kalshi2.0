from ib_async import IB, Index
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=2)

gvz = Index('GVZ', 'CBOE')
ib.qualifyContracts(gvz)

bars = ib.reqHistoricalData(
    gvz,
    endDateTime='',
    durationStr='6 M',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1,
)

if bars:
    rows = []
    for bar in bars:
        rows.append({
            "date": str(bar.date),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "gvz_vol": bar.close / 100,
        })

    df = pd.DataFrame(rows)
    df.to_csv("gvz_daily.csv", index=False)
    print(f"Saved {len(df)} bars")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.tail(10))
else:
    print("No data — may need CBOE market data subscription")

ib.disconnect()