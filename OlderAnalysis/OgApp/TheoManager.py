from IBKRMd import IBKRMd
from KalshiMd import KalshiMd
from ib_async import Ticker
import threading
import asyncio
import time
from TheoCalculator import TheoCalculator
from Contract import Contract
from datetime import datetime
from VolatilityManager import VolatilityManager
import math


class TheoManager:

    def __init__(self, target_date: str, kalshi_md: KalshiMd):
        """
        target_date: format like '26APR30' to match Kalshi ticker dates
        kalshi_md: pre-initialized KalshiMd with markets already fetched
        """
        self.target_date = target_date
        self.underlying_md = IBKRMd(self.handle_underlying_update)
        self.kalshi_md = kalshi_md
        self.kalshi_md.callback = self.handle_kalshi_update
        self.theo_calculator = TheoCalculator()
        self.volatility_manager = VolatilityManager()

        self.vol_wait_seconds = 30
        self.underlying_price = 0
        self.first_iteration = True
        self.contracts: list[Contract] = []

        # store theo results keyed by ticker
        # { "KXGOLDD-26APR0817-T4661": { "strike": ..., "bid": ..., "ask": ..., "iv": ..., "vol": ..., "theo": ..., "edge": ... } }
        self.theo_results: dict[str, dict] = {}

    def handle_underlying_update(self, tickers: set[Ticker]):
        for ticker in tickers:
            self.underlying_price = (ticker.bid + ticker.ask) / 2
            # reprice all contracts on every tick
            self._reprice_all("UND")

    def handle_kalshi_update(self, ticker: str, top: dict):
        pass

    # recalculate theo for every contract for this date and print
    def _reprice_all(self, source: str):
        if not self.contracts or self.underlying_price <= 0:
            return
        # wait until vol surface has been fitted at least once
        if not self.volatility_manager.call_fitted and not self.volatility_manager.put_fitted:
            return

        # get ALL contracts for the target date
        all_contracts = self._get_contracts_for_date(n=999)

        print(f"\n--- {source} | underlying: {self.underlying_price:.2f} ---")
        for contract in all_contracts:
            vol = .346 #self.volatility_manager.get_volatility(contract)
            if vol and vol == vol:
                theo = self.theo_calculator.calculate(self.underlying_price, vol, contract)
                mid = (contract.bestBid + contract.bestOffer) / 2
                edge = theo - mid

                # check if this contract has a raw IV from the smile fitting
                smile_contract = self.volatility_manager.call_impliedVol.get(contract.strike) or \
                                 self.volatility_manager.put_impliedVol.get(contract.strike)
                iv = smile_contract.implied_vol if smile_contract and hasattr(smile_contract, 'implied_vol') else None
                iv_str = f"{iv:.4f}" if iv and iv == iv else "---"

                # store results
                self.theo_results[contract.ticker] = {
                    "strike": contract.strike,
                    "bid": contract.bestBid,
                    "ask": contract.bestOffer,
                    "mid": mid,
                    "iv": iv,
                    "vol": vol,
                    "theo": theo,
                    "edge": edge,
                }

                print(f"{contract.ticker} | strike: {contract.strike} | bid: {contract.bestBid} | ask: {contract.bestOffer} | iv: {iv_str} | vol: {vol:.4f} | theo: {theo:.4f} | edge: {edge:.4f}")

    def _start_md(self):
        self.underlying_md.connect()
        self.underlying_md.subscribe()
        self.underlying_md.run()

    def _start_kalshi_md(self):
        async def _run():
            tickers = [m["ticker"] for m in self.kalshi_md.markets]
            print(f"Subscribing to {len(tickers)} gold markets via websocket...")
            await self.kalshi_md.connect()
            await self.kalshi_md.subscribe_all(tickers)
            await self.kalshi_md.listen()

        asyncio.run(_run())

    def _get_contracts_for_date(self, n: int = 5) -> list[Contract]:
        matching = {}
        for date_key, contracts in self.kalshi_md.books.items():
            if date_key.startswith(self.target_date):
                matching.update(contracts)

        if not matching or self.underlying_price <= 0:
            return []

        contracts_with_strikes = []
        for ticker, book in matching.items():
            parts = ticker.split("-T")
            if len(parts) < 2:
                continue
            try:
                strike = float(parts[1])
            except ValueError:
                continue
            contracts_with_strikes.append((ticker, strike, book))

        # sort by strike low to high
        contracts_with_strikes.sort(key=lambda x: x[1])

        if n >= len(contracts_with_strikes):
            # return all if requesting more than available
            selected = contracts_with_strikes
        else:
            # split into ITM, ATM, OTM
            itm = [c for c in contracts_with_strikes if c[1] < self.underlying_price]
            otm = [c for c in contracts_with_strikes if c[1] > self.underlying_price]
            atm = [c for c in contracts_with_strikes if c[1] == self.underlying_price]

            # if no exact ATM, take closest
            if not atm:
                closest = min(contracts_with_strikes, key=lambda x: abs(x[1] - self.underlying_price))
                atm = [closest]
                itm = [c for c in itm if c[0] != closest[0]]
                otm = [c for c in otm if c[0] != closest[0]]

            # evenly sample from each bucket
            atm_count = 1
            side_count = (n - atm_count) // 2

            # spread evenly across ITM and OTM ranges
            selected_itm = self._sample_spread(itm, side_count)
            selected_otm = self._sample_spread(otm, side_count)

            selected = selected_itm + atm + selected_otm

        result = []
        for ticker, strike, book in selected:
            date_str = self.kalshi_md._parse_date(ticker)
            hour = int(date_str[-2:]) if len(date_str) >= 2 else 17
            month_map = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            try:
                year = 2000 + int(date_str[:2])
                month = month_map[date_str[2:5]]
                day = int(date_str[5:7])
                expiration = datetime(year, month, day, hour, 0)
            except (ValueError, KeyError):
                continue

            contract = Contract("call", strike, expiration)
            # store ticker on contract for printing
            contract.ticker = ticker

            yes_bid = book["best_yes_bid"]
            no_bid = book["best_no_bid"]

            contract.bestBid = yes_bid[0] if yes_bid else 0
            # yes ask = 1 - best no bid
            contract.bestOffer = round(1.0 - no_bid[0], 2) if no_bid else 0

            # skip contracts with no valid two-sided market
            if contract.bestBid <= 0 or contract.bestOffer <= 0:
                continue

            result.append(contract)

        return result

    # pick n evenly spaced items from a sorted list
    def _sample_spread(self, items: list, n: int) -> list:
        if n <= 0 or not items:
            return []
        if len(items) <= n:
            return items
        # evenly spaced indices
        step = (len(items) - 1) / (n - 1) if n > 1 else 0
        indices = [round(i * step) for i in range(n)]
        return [items[i] for i in indices]

    def _start_volatility_calculation(self):
        # wait for both data sources to initialize
        while self.underlying_price <= 0 or not self.kalshi_md.books:
            time.sleep(1)

        while True:
            # refresh smile contracts from live book
            self.contracts = self._get_contracts_for_date(n=10)
            for contract in self.contracts:
                self.volatility_manager.update_contract(contract)

            if self.contracts:
                # refit vol surface
                self.volatility_manager.recalculate(self.underlying_price)
                # reprice all contracts after refit
                self._reprice_all("VOL")

            time.sleep(self.vol_wait_seconds)

    def run(self):
        self.underlying_thread = threading.Thread(target=self._start_md, daemon=True)
        self.underlying_thread.start()

        self.kalshi_thread = threading.Thread(target=self._start_kalshi_md, daemon=True)
        self.kalshi_thread.start()

        self.vol_thread = threading.Thread(target=self._start_volatility_calculation, daemon=True)
        self.vol_thread.start()


# fetch markets via REST before anything starts
kalshi_md = KalshiMd(callback=lambda t, top: None)

daily = kalshi_md.fetch_markets("KXGOLDD")
monthly = kalshi_md.fetch_markets("KXGOLDMON")

# combine both
kalshi_md.markets = daily + monthly

if not kalshi_md.markets:
    print("No gold markets found.")
    exit(1)

# parse and display available dates
dates = set()
for m in kalshi_md.markets:
    parts = m["ticker"].split("-")
    if len(parts) >= 2:
        dates.add(parts[1][:7])

print(f"\nAvailable dates: {sorted(dates)}")
print(f"Total daily: {len(daily)} | Total monthly: {len(monthly)} | Combined: {len(kalshi_md.markets)}")

# prompt for target date
target = input("\nEnter target date (e.g. 26APR30): ").strip()

# pass pre-fetched kalshi_md into TheoManager
manager = TheoManager(target_date=target, kalshi_md=kalshi_md)
manager.run()

while True:
    time.sleep(2)
    print("here")