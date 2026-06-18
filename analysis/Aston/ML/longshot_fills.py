'''
Overview: 
    Grab all fills in the range where we want to trade. If theo supports it trade it (some amount of edge).
    Check if the mean c/fill is significant
'''


import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import load_trades, load_theo, fetch_settlements_from_api, secs_to_expiry


'''
Graph daily pnl mean by edge to see which is visually optimal
'''
def edge_test(tt):
    #now iterate through many edge levels and graph the mean pnl results 
    results = []
    for i in np.arange(-0.1, 0.2, .01):
        needed_edge = i
        tt_edge_check = tt[tt['edge'] >= needed_edge].copy() 
        tt_edge_check['moneyness'] = tt_edge_check['yes_price'] - tt_edge_check['outcome']

        tt_edge_check['secs_to_exp'] = secs_to_expiry(tt_edge_check['ticker'], tt_edge_check['ts'])
        tt_edge_check = tt_edge_check[tt_edge_check['secs_to_exp'] >= 90] #trading only in first 14.5 mins

        #split by day since we can assume the days are independent
        tt_edge_check['date'] = pd.to_datetime(tt_edge_check['ts'], utc=True).dt.date
        tt_day_check = tt_edge_check.groupby('date').agg(day_pnl=('moneyness', 'sum'), c_per_fill=('moneyness', 'mean'), trades_count=('moneyness', 'count'))


        per_mkt = tt_edge_check.groupby('ticker').size()   # trades in each market
        mean_per_mkt = per_mkt.mean()
        std_per_mkt  = per_mkt.std(ddof=1)

        print("Edge Level: " , needed_edge , "Avg Trades Per Market (15 min): ", mean_per_mkt , " Std-Dev of Trades Per Market: ", std_per_mkt )

        daily_trade_count_mean = tt_day_check['trades_count'].mean()        
        c_per_fill_mean = tt_day_check['c_per_fill'].mean()
        results.append((i, c_per_fill_mean, daily_trade_count_mean))


    d = results
    edges    = [d[0] for d in results]
    mean_pnl = [d[1] for d in results]
    counts   = [d[2] for d in results]

    plt.scatter(edges, mean_pnl)
    for x, y, c in zip(edges, mean_pnl, counts):
        plt.annotate(f"{c:,.0f}", (x, y),
                    textcoords="offset points", xytext=(0, 6),
                    ha='center', fontsize=8)

    plt.xlabel('Edge level')
    plt.ylabel('c/fill PnL')
    plt.title('Mean PnL by edge threshold (label = trade count)')
    plt.tight_layout()
    plt.show()



if Path('./longshot_results.csv').exists():
    tt = pd.read_csv('./longshot_results.csv')
else:
    date = '2026-5-16'
    trades = load_trades(date)
    theos = load_theo(date, until='today')

    trades = trades.sort_values(by='ts')
    theos = theos.sort_values(by='ts')

    mask = (trades['yes_price'] >= 0.1) & (trades['yes_price'] <= 0.35)
    trades = trades[mask]

    tt = pd.merge_asof(left=trades, right=theos[['ts', 'theo', 'ticker', 'spot']], on='ts', by='ticker', suffixes=['_tr', '_t'], direction='backward')

    cache = Path(__file__).resolve().parent / ".settlements_cache.json"
    settlements = fetch_settlements_from_api(tickers=tt['ticker'].unique().tolist(), cache_path=cache)
    tt['outcome'] = tt['ticker'].map(settlements)

    tt['edge'] = tt['yes_price'] - tt['theo']

    #save to csv
    tt.to_csv('./data/longshot_results.csv')

#calculate the trades I would have wanted to be apart of
needed_edge = 0.04
tt_edge = tt[tt['edge'] >= needed_edge].copy() 
tt_edge['moneyness'] = tt_edge['yes_price'] - tt_edge['outcome']

#sum total amount
total_pnl = np.sum(tt_edge['moneyness'])
cents_per_fill = total_pnl / len(tt_edge)

#time til expiration
tt_edge['secs_to_exp'] = secs_to_expiry(tt_edge['ticker'], tt_edge['ts'])
print("percent in last 60 seconds: ", (len(tt_edge[tt_edge['secs_to_exp'] <= 90])/ len(tt_edge)))
tt_edge = tt_edge[tt_edge['secs_to_exp'] >= 90] #trading only in first 14.5 mins

#run a one sample t-test to test the significants of the findings
t, p = stats.ttest_1samp(tt_edge['moneyness'].dropna(), popmean=0)   # two-sided
total_pnl = np.sum(tt_edge['moneyness'])
cents_per_fill = total_pnl / len(tt_edge)

print('Edge Level', needed_edge ,'Total Pnl:', total_pnl, ' c/fill:', cents_per_fill, 'Total Trades:', len(tt_edge))
print('Scipy T_stat:', t, ' p-value:', p)

#split by day since we can assume the days are independent
tt_edge['date'] = pd.to_datetime(tt_edge['ts'], utc=True).dt.date
tt_day = tt_edge.groupby('date').agg(day_pnl=('moneyness', 'sum'), c_per_fill=('moneyness', 'mean'), trades_count=('moneyness', 'count'))



t_day, p_day = stats.ttest_1samp(tt_day['day_pnl'].dropna(), popmean=0)
day_mean = tt_day['day_pnl'].mean()
c_per_fill_mean = tt_day['c_per_fill'].mean()
trade_count_mean = tt_day['trades_count'].mean()
print('Day T-Stat:', t_day, ' p-value:', p_day, ' daily_pnl_mean:', day_mean, ' c_per_fill_mean:', c_per_fill_mean, ' Mean Day Trade Count:', trade_count_mean)

total_markets_count = (tt_edge['date'].max() - tt_edge['date'].min()).days * 24 * 4
market_participate_in = len(tt_edge['ticker'].unique())
percentage_markets_participate_in =  round(market_participate_in / total_markets_count * 100, 2)
print("Total Markets: ", total_markets_count, ' Markets Participate In: ', market_participate_in, ' Percent:', percentage_markets_participate_in, '%')

#edge_test(tt)

# start with edge level of 0.04 and regress to see if its just noise or there's edge
tt_edge ['ts'] = pd.to_datetime(tt_edge['ts'], utc=True)
at8 = (tt_edge[tt_edge['ts'].dt.hour >= 8]
        .sort_values('ts')
        .groupby('date', as_index=False)
        .first())

at8['log_ret'] = np.log(at8['spot']).diff()   # log(today_8am) − log(yesterday_8am)

#add in the log returns
tt_day = pd.merge(left=tt_day, right=at8[['log_ret', 'date']], on='date')
tt_day['log_ret'] = tt_day['log_ret'].fillna(0)


#calculate and graph ols for pnl vs log_ret of underlying
m = smf.ols("day_pnl ~ log_ret", data=tt_day).fit()
print(m.summary())

d = tt_day.dropna(subset=['log_ret', 'day_pnl'])

plt.scatter(d['log_ret'], d['day_pnl'])

# fitted line: intercept + slope * x
x = np.linspace(d['log_ret'].min(), d['log_ret'].max(), 100)
y = m.params['Intercept'] + m.params['log_ret'] * x
plt.plot(x, y, color='red')

plt.axvline(0, color='gray', lw=0.8)   # ETH flat — line crosses here = the intercept
plt.axhline(0, color='gray', lw=0.8)
plt.xlabel('Daily ETH log return')
plt.ylabel('Day PnL')
plt.title(f"intercept={m.params['Intercept']:.1f} (p={m.pvalues['Intercept']:.3f}), "
        f"slope={m.params['log_ret']:.0f}")
plt.show()