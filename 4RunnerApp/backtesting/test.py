import pandas as pd
df = pd.read_csv("../data/kxbtc_weekly_bracket_trades.csv")
print(df.columns.tolist())
print(df.head(3).to_string())
print(f"\nSample yes_sub_title: {df['yes_sub_title'].iloc[0]}")
print(f"Sample settlement_result: {df['settlement_result'].unique()}")
print(f"Sample ticker: {df['ticker'].iloc[0]}")