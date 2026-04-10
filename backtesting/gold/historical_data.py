import pandas as pd

df = pd.read_csv("kxgoldmon_trades.csv")
print(f"Total trades: {len(df)}")
print(f"Unique tickers: {df['ticker'].nunique()}")
print(f"Date range: {df['created_time'].min()} to {df['created_time'].max()}")
print(f"\nTrades per ticker (top 20):")
print(df['ticker'].value_counts().head(20))
print(f"\nTrades per strike:")
print(df.groupby('strike').size().describe())