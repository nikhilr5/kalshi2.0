from ib_async import IB, FuturesOption
import time

ib = IB()
ib.connect('127.0.0.1', 4001, clientId=2)

# request delayed data
ib.reqMarketDataType(3)

fop = FuturesOption(symbol='GC', exchange='COMEX', right='C', lastTradeDateOrContractMonth='20260427')
details = ib.reqContractDetails(fop)

near_atm = [d for d in details if 4600 <= d.contract.strike <= 4800]
near_atm.sort(key=lambda d: d.contract.strike)

print(f"Found {len(near_atm)} contracts near ATM")

tickers = []
for d in near_atm[:10]:
    ticker = ib.reqMktData(d.contract, genericTickList='106')
    tickers.append(ticker)

ib.sleep(10)

print(f"\n{'Strike':<10} {'Bid':<8} {'Ask':<8} {'IV':<10} {'Delta':<10}")
print("-" * 50)
for ticker in tickers:
    strike = ticker.contract.strike
    bid = f"{ticker.bid:.2f}" if ticker.bid and ticker.bid > 0 else '---'
    ask = f"{ticker.ask:.2f}" if ticker.ask and ticker.ask > 0 else '---'
    greeks = ticker.modelGreeks
    if greeks and greeks.impliedVol:
        iv = f"{greeks.impliedVol:.4f}"
        delta = f"{greeks.delta:.4f}" if greeks.delta else '---'
    else:
        iv = '---'
        delta = '---'
    print(f"{strike:<10} {bid:<8} {ask:<8} {iv:<10} {delta:<10}")

ib.disconnect()