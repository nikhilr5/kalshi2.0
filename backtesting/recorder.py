"""
Market data + trade recorder for Kalshi KXBTCD above/below markets.

Records:
    1. Every fill (trade) on your account with market context at fill time
    2. Periodic snapshots (every 5s) of Kalshi bid/ask + computed theo
       for all markets within 8% OTM

Stores to SQLite (marketdata/recorder.db) and can export to Parquet.

Usage:
    python recorder.py                  # record live data
    python recorder.py --export 2026-04-17  # export a day to parquet
"""

import sys
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

# Add 4RunnerApp2.0 to path for shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "4RunnerApp2.0"))

from kalshi_api import KalshiAPI
from market_discovery import discover_events_for_series, parse_strike, display_strike
from btc_price_feed import CryptoPriceFeed
from ws_feed import KalshiWsFeed
from deribit_vol import (
    DeribitBracketPricer, DeribitWsFeed, find_deribit_expiry,
    find_weekly_deribit_expiry, KALSHI_TO_DERIBIT_CURRENCY,
)
from db import RecorderDB


SERIES_LIST = ["KXBTCD", "KXBTC"]  # daily + weekly above/below
COINBASE_PRODUCT = "BTC-USD"
DERIBIT_CURRENCY = "BTC"
OTM_FILTER_PCT = 8.0
SNAPSHOT_INTERVAL = 5  # seconds
STALE_THRESHOLD = 30  # seconds — feeds stale after this long with no data


class Recorder:

    def __init__(self):
        self.api = KalshiAPI()
        self.db = RecorderDB()

        self.running = False
        self.spot_price = 0.0
        self.spot_bid = 0.0
        self.spot_ask = 0.0

        self.price_feed = None
        self.ws_feed = None

        # Per-expiry Deribit pricers and WS feeds
        # {deribit_expiry_str: DeribitBracketPricer}
        self.pricers = {}
        # {deribit_expiry_str: DeribitWsFeed}
        self.deribit_feeds = {}
        # {event_close_time: deribit_expiry_str} — maps each event's close to its Deribit expiry
        self.close_to_expiry = {}

        # Weekly (Friday) Deribit pricer — used for all daily events as alternative theo
        self.weekly_pricer = None       # DeribitBracketPricer
        self.weekly_deribit_feed = None # DeribitWsFeed
        self.weekly_expiry_str = None   # e.g. "25APR26"

        # Markets we're tracking: {ticker: {display_strike, event_ticker, close_time}}
        self.tracked = {}

        # Kalshi book state from WS: {ticker: {yes_bid, yes_ask, bid_size, ask_size}}
        self.book = {}

        # Cached theos: {raw_strike: (bid_theo, ask_theo)}
        self.theos = {}

        self.session_id = None
        self.event_ticker = ""

    def _get_pricer(self, close_time: str):
        """Get the DeribitBracketPricer for a given event close_time."""
        expiry = self.close_to_expiry.get(close_time)
        if expiry and expiry in self.pricers:
            return self.pricers[expiry]
        return None

    def _start_weekly_deribit(self):
        """Start a Deribit pricer + WS for the nearest Friday expiry (weekly)."""
        expiry_str = find_weekly_deribit_expiry(DERIBIT_CURRENCY)
        if not expiry_str:
            print("[Recorder] No weekly (Friday) Deribit expiry found")
            return
        if expiry_str == self.weekly_expiry_str and self.weekly_pricer:
            return  # already connected

        # If the weekly expiry is the same as an existing per-event pricer, reuse it
        if expiry_str in self.pricers:
            self.weekly_pricer = self.pricers[expiry_str]
            self.weekly_expiry_str = expiry_str
            print(f"[Recorder] Weekly pricer reusing existing {expiry_str}")
            return

        self.weekly_expiry_str = expiry_str
        pricer = DeribitBracketPricer()
        pricer.currency = DERIBIT_CURRENCY
        pricer.risk_free_rate = 0.043
        self.weekly_pricer = pricer

        print(f"[Recorder] Discovering weekly Deribit {expiry_str}...")
        instruments = pricer.discover_instruments(expiry_str)
        if instruments:
            feed = DeribitWsFeed(pricer, self._on_deribit_update)
            feed.start(instruments)
            self.weekly_deribit_feed = feed
            print(f"[Recorder] Weekly Deribit WS started for {expiry_str}: {len(instruments)} options")
        else:
            print(f"[Recorder] No weekly Deribit instruments found for {expiry_str}")

    def _start_deribit_for_expiry(self, close_time: str):
        """Start a Deribit pricer + WS feed for a given event close_time if not already running."""
        if close_time in self.close_to_expiry:
            return  # already set up

        expiry_str = find_deribit_expiry(close_time, currency=DERIBIT_CURRENCY)
        if not expiry_str:
            print(f"[Recorder] No Deribit expiry match for close={close_time}")
            return

        self.close_to_expiry[close_time] = expiry_str

        if expiry_str in self.pricers:
            # Already have a pricer for this Deribit expiry (shared by multiple events)
            return

        pricer = DeribitBracketPricer()
        pricer.currency = DERIBIT_CURRENCY
        pricer.risk_free_rate = 0.043
        self.pricers[expiry_str] = pricer

        print(f"[Recorder] Discovering Deribit {expiry_str} (for close={close_time})...")
        instruments = pricer.discover_instruments(expiry_str)
        if instruments:
            feed = DeribitWsFeed(pricer, self._on_deribit_update)
            feed.start(instruments)
            self.deribit_feeds[expiry_str] = feed
            print(f"[Recorder] Deribit WS started for {expiry_str}: {len(instruments)} options")
        else:
            print(f"[Recorder] No Deribit instruments found for {expiry_str}")

    def start(self):
        self.running = True
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        # Discover events across all series (daily + weekly)
        all_markets = {}  # ticker -> info
        for series in SERIES_LIST:
            print(f"[Recorder] Discovering {series} events...")
            events = discover_events_for_series(self.api, series)
            for event in events:
                et = event["event_ticker"]
                close = event.get("close_time", "")
                for m in event["markets"]:
                    ticker = m["ticker"]
                    raw = parse_strike(ticker)
                    if raw > 0 and ticker not in all_markets:
                        all_markets[ticker] = {
                            "display_strike": display_strike(raw),
                            "event_ticker": et,
                            "close_time": close,
                        }
                print(f"[Recorder]   {et}: {len(event['markets'])} markets")

        if not all_markets:
            print("[Recorder] No events found")
            return

        # Start Coinbase feed
        print(f"[Recorder] Starting Coinbase feed for {COINBASE_PRODUCT}...")
        self.price_feed = CryptoPriceFeed(self._on_price, COINBASE_PRODUCT)
        self.price_feed.start()

        # Wait for first spot price
        print("[Recorder] Waiting for spot price...")
        for _ in range(100):
            if self.spot_price > 0:
                break
            time.sleep(0.1)
        if self.spot_price <= 0:
            print("[Recorder] No spot price received, using all markets")
            self.tracked = all_markets
        else:
            # Filter to within OTM_FILTER_PCT
            for ticker, info in all_markets.items():
                disp = info["display_strike"]
                otm = abs((disp - self.spot_price) / self.spot_price * 100)
                if otm <= OTM_FILTER_PCT:
                    self.tracked[ticker] = info

        print(f"[Recorder] Tracking {len(self.tracked)} markets "
              f"(within {OTM_FILTER_PCT}% OTM of ${self.spot_price:,.0f})")

        # Start Kalshi WS feed
        tickers = list(self.tracked.keys())
        if tickers:
            self.ws_feed = KalshiWsFeed(self.api, self._on_ws_update)
            self.ws_feed.start(tickers)
            print(f"[Recorder] Kalshi WS started for {len(tickers)} tickers")

        # Start per-expiry Deribit WS feeds
        close_times = set(info["close_time"] for info in self.tracked.values() if info["close_time"])
        for ct in sorted(close_times):
            self._start_deribit_for_expiry(ct)

        # Start weekly (Friday) Deribit feed for alternative theo
        self._start_weekly_deribit()

        # Log session
        series_str = "+".join(SERIES_LIST)
        event_tickers = sorted(set(info["event_ticker"] for info in self.tracked.values()))
        self.session_id = self.db.insert_session(
            series_str, ",".join(event_tickers), len(self.tracked),
            SNAPSHOT_INTERVAL, OTM_FILTER_PCT,
        )
        print(f"[Recorder] Session {self.session_id} started")

        # Seed known fills so we don't re-record old ones
        self._seen_fill_ids = set()
        self._seed_known_fills()

        # Main snapshot loop
        self._run_loop()

    def _seed_known_fills(self):
        """Fetch all existing fills from Kalshi and mark them as seen
        so the poller only records new ones going forward."""
        try:
            fills = self.api.get_fills()
            for f in fills:
                fid = f.get("trade_id") or f.get("fill_id") or f.get("id", "")
                if fid:
                    self._seen_fill_ids.add(str(fid))
            print(f"[Recorder] Seeded {len(self._seen_fill_ids)} existing fills")
        except Exception as e:
            print(f"[Recorder] Seed fills failed: {e}")

    def _poll_fills(self):
        """Poll REST API for new fills and record any we haven't seen."""
        try:
            fills = self.api.get_fills()
        except Exception as e:
            print(f"[Recorder] Fill poll failed: {e}")
            return

        spot_b = self.spot_bid if self.spot_bid > 0 else self.spot_price
        spot_a = self.spot_ask if self.spot_ask > 0 else self.spot_price

        for f in fills:
            fid = str(f.get("trade_id") or f.get("fill_id") or f.get("id", ""))
            if not fid or fid in self._seen_fill_ids:
                continue
            self._seen_fill_ids.add(fid)

            ticker = f.get("ticker", "")
            action = f.get("action", "")
            side = f.get("side", "yes")
            count = float(f.get("count_fp", f.get("count", 0)))
            price = float(f.get("yes_price_dollars", 0) or f.get("yes_price", 0) or 0)

            # Normalize to yes terms: no-side buy→sell, no-side sell→buy
            if side == "no":
                action = "sell" if action == "buy" else "buy"
                side = "yes"
            fee = float(f.get("fee_cost", 0) or 0)

            # Look up strike info
            info = self.tracked.get(ticker, {})
            strike = info.get("display_strike", 0.0)
            event_ticker = info.get("event_ticker", "")
            close = info.get("close_time", "")

            # Compute theos at fill time
            theo_bid = 0.0
            theo_ask = 0.0
            deribit_bid_iv = 0.0
            deribit_ask_iv = 0.0
            bk = self.book.get(ticker, {})
            pricer = self._get_pricer(close)
            if strike > 0 and pricer and pricer.options and spot_b > 0:
                t_bid = pricer.prob_above_bid_iv(strike, spot=spot_b, kalshi_close_iso=close)
                t_ask = pricer.prob_above_ask_iv(strike, spot=spot_a, kalshi_close_iso=close)
                theo_bid = min(t_bid, t_ask)
                theo_ask = max(t_bid, t_ask)
                deribit_bid_iv = pricer._find_closest_bid_iv(strike)
                deribit_ask_iv = pricer._find_closest_ask_iv(strike)

            # Weekly theo at fill time
            theo_bid_w = 0.0
            theo_ask_w = 0.0
            db_bid_iv_w = 0.0
            db_ask_iv_w = 0.0
            wp = self.weekly_pricer
            if strike > 0 and wp and wp.options and spot_b > 0:
                wb_iv = wp._find_closest_bid_iv(strike)
                wa_iv = wp._find_closest_ask_iv(strike)
                if wb_iv > 0:
                    theo_bid_w = wp.prob_above_with_iv(strike, wb_iv, spot=spot_b, kalshi_close_iso=close)
                if wa_iv > 0:
                    theo_ask_w = wp.prob_above_with_iv(strike, wa_iv, spot=spot_a, kalshi_close_iso=close)
                db_bid_iv_w = wb_iv
                db_ask_iv_w = wa_iv

            client_order_id = f.get("client_order_id", "")

            self.db.insert_fill(
                ticker=ticker, action=action, side=side,
                count=count, price=price, strike=strike,
                event_ticker=event_ticker,
                spot_bid=spot_b, spot_ask=spot_a,
                theo_bid=theo_bid, theo_ask=theo_ask,
                kalshi_yes_bid=bk.get("yes_bid", 0),
                kalshi_yes_ask=bk.get("yes_ask", 0),
                deribit_bid_iv=deribit_bid_iv,
                deribit_ask_iv=deribit_ask_iv,
                client_order_id=client_order_id,
                theo_bid_weekly=theo_bid_w, theo_ask_weekly=theo_ask_w,
                deribit_bid_iv_weekly=db_bid_iv_w, deribit_ask_iv_weekly=db_ask_iv_w,
                fee=fee,
            )
            orig_side = f.get("side", "yes")
            tag = "init" if client_order_id.startswith("init_") else \
                  "flat" if client_order_id.startswith("flat_") else "?"
            fee_str = f" fee=${fee:.2f}" if fee > 0 else ""
            norm_str = f" (from {orig_side})" if orig_side == "no" else ""
            print(f"[Recorder] FILL (REST): {ticker} {action} {side} x{count} "
                  f"@ ${price:.2f} [{tag}]{fee_str}{norm_str} (theo={theo_bid:.3f}/{theo_ask:.3f} "
                  f"wk={theo_bid_w:.3f}/{theo_ask_w:.3f})")

    def _check_staleness(self):
        """Check all feeds for staleness. Log warnings and reconnect if needed."""
        now = time.time()
        stale_feeds = []

        # Check Coinbase price feed
        if self.price_feed and self.price_feed.last_update_ts > 0:
            age = now - self.price_feed.last_update_ts
            if age > STALE_THRESHOLD:
                stale_feeds.append(f"Coinbase ({age:.0f}s)")

        # Check Kalshi WS feed
        if self.ws_feed and self.ws_feed.last_update_ts > 0:
            age = now - self.ws_feed.last_update_ts
            if age > STALE_THRESHOLD:
                stale_feeds.append(f"Kalshi WS ({age:.0f}s)")

        # Check per-expiry Deribit feeds
        for exp_str, feed in self.deribit_feeds.items():
            if feed.last_update_ts > 0:
                age = now - feed.last_update_ts
                if age > STALE_THRESHOLD:
                    stale_feeds.append(f"Deribit {exp_str} ({age:.0f}s)")

        # Check weekly Deribit feed
        if self.weekly_deribit_feed and self.weekly_deribit_feed.last_update_ts > 0:
            age = now - self.weekly_deribit_feed.last_update_ts
            if age > STALE_THRESHOLD:
                stale_feeds.append(f"Deribit weekly ({age:.0f}s)")

        if stale_feeds:
            print(f"[Recorder] STALE feeds: {', '.join(stale_feeds)}")

        return len(stale_feeds) > 0

    def _run_loop(self):
        last_snapshot = 0
        last_refilter = 0
        last_fill_poll = 0
        last_stale_check = 0

        while self.running:
            time.sleep(0.5)
            now = time.time()

            # Poll fills every 5s
            if now - last_fill_poll >= 5:
                last_fill_poll = now
                self._poll_fills()

            # Check feed staleness every 15s
            if now - last_stale_check >= 15:
                last_stale_check = now
                self._check_staleness()

            # Refilter markets every 60s as spot moves
            if now - last_refilter >= 60:
                last_refilter = now
                self._refilter_markets()

            # Snapshot every SNAPSHOT_INTERVAL
            if now - last_snapshot >= SNAPSHOT_INTERVAL:
                last_snapshot = now
                self._take_snapshot()

    def _refilter_markets(self):
        """Pick up new events/markets, drop expired ones. Handles event transitions.
        Discovers across all series in SERIES_LIST."""
        if self.spot_price <= 0:
            return
        try:
            now = datetime.now(tz=timezone.utc)
            new_tickers = []

            for series in SERIES_LIST:
                events = discover_events_for_series(self.api, series)
                if not events:
                    continue

                for event in events:
                    et = event["event_ticker"]
                    close_str = event.get("close_time", "")

                    # Skip events that closed more than 10 min ago
                    if close_str:
                        try:
                            close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                            if (now - close_utc).total_seconds() > 600:
                                continue
                        except Exception:
                            pass

                    for m in event["markets"]:
                        ticker = m["ticker"]
                        if ticker in self.tracked:
                            continue
                        raw = parse_strike(ticker)
                        if raw <= 0:
                            continue
                        disp = display_strike(raw)
                        otm = abs((disp - self.spot_price) / self.spot_price * 100)
                        if otm <= OTM_FILTER_PCT:
                            self.tracked[ticker] = {
                                "display_strike": disp,
                                "event_ticker": et,
                                "close_time": close_str,
                            }
                            new_tickers.append(ticker)

            # Drop markets whose event expired > 10 min ago
            expired = []
            for ticker, info in self.tracked.items():
                close_str = info.get("close_time", "")
                if close_str:
                    try:
                        close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        if (now - close_utc).total_seconds() > 600:
                            expired.append(ticker)
                    except Exception:
                        pass
            for ticker in expired:
                del self.tracked[ticker]

            if expired:
                print(f"[Recorder] Dropped {len(expired)} expired markets")

            if new_tickers and self.ws_feed:
                self.ws_feed.subscribe_tickers(new_tickers)
                print(f"[Recorder] Added {len(new_tickers)} new markets "
                      f"(total: {len(self.tracked)})")

            # Start Deribit feeds for any new close times
            current_close_times = set(info["close_time"] for info in self.tracked.values() if info["close_time"])
            for ct in sorted(current_close_times):
                self._start_deribit_for_expiry(ct)

            # Refresh weekly Deribit if needed (e.g. crossed into new week)
            self._start_weekly_deribit()

            # Clean up close_to_expiry for expired close times
            for ct in list(self.close_to_expiry.keys()):
                if ct not in current_close_times:
                    del self.close_to_expiry[ct]

            # Stop Deribit feeds for expiries no longer needed
            needed_expiries = set(self.close_to_expiry.values())
            for exp_str in list(self.deribit_feeds.keys()):
                if exp_str not in needed_expiries:
                    print(f"[Recorder] Stopping Deribit WS for expired {exp_str}")
                    self.deribit_feeds[exp_str].stop()
                    del self.deribit_feeds[exp_str]
                    if exp_str in self.pricers:
                        del self.pricers[exp_str]

        except Exception as e:
            print(f"[Recorder] Refilter error: {e}")

    def _take_snapshot(self):
        """Snapshot all tracked markets with current theos."""
        if not self.tracked:
            return

        spot_b = self.spot_bid if self.spot_bid > 0 else self.spot_price
        spot_a = self.spot_ask if self.spot_ask > 0 else self.spot_price
        spot_mid = (spot_b + spot_a) / 2 if spot_b > 0 and spot_a > 0 else self.spot_price

        rows = []
        for ticker, info in self.tracked.items():
            disp = info["display_strike"]
            close = info.get("close_time", "")

            # Kalshi book
            bk = self.book.get(ticker, {})
            yes_bid = bk.get("yes_bid", 0)
            yes_ask = bk.get("yes_ask", 0)
            bid_size = bk.get("bid_size", 0)
            ask_size = bk.get("ask_size", 0)

            # Compute theo using the correct pricer for this event's expiry
            theo_bid = 0.0
            theo_ask = 0.0
            deribit_bid_iv = 0.0
            deribit_ask_iv = 0.0
            pricer = self._get_pricer(close)
            if pricer and pricer.options and spot_b > 0 and spot_a > 0:
                t_bid = pricer.prob_above_bid_iv(disp, spot=spot_b, kalshi_close_iso=close)
                t_ask = pricer.prob_above_ask_iv(disp, spot=spot_a, kalshi_close_iso=close)
                theo_bid = t_bid
                theo_ask = t_ask
                deribit_bid_iv = pricer._find_closest_bid_iv(disp)
                deribit_ask_iv = pricer._find_closest_ask_iv(disp)

            # Weekly theo: use weekly Deribit IVs but event's close_time for T
            theo_bid_w = 0.0
            theo_ask_w = 0.0
            db_bid_iv_w = 0.0
            db_ask_iv_w = 0.0
            wp = self.weekly_pricer
            if wp and wp.options and spot_b > 0 and spot_a > 0:
                wb_iv = wp._find_closest_bid_iv(disp)
                wa_iv = wp._find_closest_ask_iv(disp)
                if wb_iv > 0:
                    theo_bid_w = wp.prob_above_with_iv(disp, wb_iv, spot=spot_b, kalshi_close_iso=close)
                if wa_iv > 0:
                    theo_ask_w = wp.prob_above_with_iv(disp, wa_iv, spot=spot_a, kalshi_close_iso=close)
                db_bid_iv_w = wb_iv
                db_ask_iv_w = wa_iv

            self.theos[ticker] = (theo_bid, theo_ask)

            otm_pct = (disp - spot_mid) / spot_mid * 100 if spot_mid > 0 else 0

            rows.append({
                "ticker": ticker,
                "event_ticker": info.get("event_ticker", ""),
                "strike": disp,
                "close_time": close,
                "kalshi_yes_bid": yes_bid,
                "kalshi_yes_ask": yes_ask,
                "kalshi_bid_size": bid_size,
                "kalshi_ask_size": ask_size,
                "spot_bid": spot_b,
                "spot_ask": spot_a,
                "spot_mid": spot_mid,
                "theo_bid": theo_bid,
                "theo_ask": theo_ask,
                "deribit_bid_iv": deribit_bid_iv,
                "deribit_ask_iv": deribit_ask_iv,
                "deribit_index": pricer.index_price if pricer else 0,
                "otm_pct": otm_pct,
                "edge_bid": yes_bid - theo_ask if yes_bid > 0 and theo_ask > 0 else 0,
                "edge_ask": theo_bid - yes_ask if theo_bid > 0 and yes_ask > 0 else 0,
                "theo_bid_weekly": theo_bid_w,
                "theo_ask_weekly": theo_ask_w,
                "deribit_bid_iv_weekly": db_bid_iv_w,
                "deribit_ask_iv_weekly": db_ask_iv_w,
            })

        self.db.insert_snapshots(rows)
        print(f"[Recorder] Snapshot: {len(rows)} markets, "
              f"spot=${spot_mid:,.2f}")

    # --- Callbacks ---

    def _on_price(self, price: float, bid: float = 0.0, ask: float = 0.0):
        self.spot_price = price
        if bid > 0:
            self.spot_bid = bid
        if ask > 0:
            self.spot_ask = ask

    def _on_ws_update(self, ticker: str, yes_bid: float, yes_ask: float,
                      bid_size: int = 0, ask_size: int = 0):
        self.book[ticker] = {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    def _on_fill(self, ticker: str, action: str, side: str,
                 price: float, count: int):
        """Record every fill with market context at fill time."""
        # Find strike info — tracked is now keyed by ticker
        info = self.tracked.get(ticker, {})
        strike = info.get("display_strike", 0.0)
        event_ticker = info.get("event_ticker", "")
        close = info.get("close_time", "")

        # Current market context
        bk = self.book.get(ticker, {})
        spot_b = self.spot_bid if self.spot_bid > 0 else self.spot_price
        spot_a = self.spot_ask if self.spot_ask > 0 else self.spot_price

        theo_bid = 0.0
        theo_ask = 0.0
        deribit_bid_iv = 0.0
        deribit_ask_iv = 0.0
        pricer = self._get_pricer(close)
        if strike > 0 and pricer and pricer.options and spot_b > 0:
            t_bid = pricer.prob_above_bid_iv(strike, spot=spot_b, kalshi_close_iso=close)
            t_ask = pricer.prob_above_ask_iv(strike, spot=spot_a, kalshi_close_iso=close)
            theo_bid = min(t_bid, t_ask)
            theo_ask = max(t_bid, t_ask)
            deribit_bid_iv = pricer._find_closest_bid_iv(strike)
            deribit_ask_iv = pricer._find_closest_ask_iv(strike)

        # Weekly theo at fill time
        theo_bid_w = 0.0
        theo_ask_w = 0.0
        db_bid_iv_w = 0.0
        db_ask_iv_w = 0.0
        wp = self.weekly_pricer
        if strike > 0 and wp and wp.options and spot_b > 0:
            wb_iv = wp._find_closest_bid_iv(strike)
            wa_iv = wp._find_closest_ask_iv(strike)
            if wb_iv > 0:
                theo_bid_w = wp.prob_above_with_iv(strike, wb_iv, spot=spot_b, kalshi_close_iso=close)
            if wa_iv > 0:
                theo_ask_w = wp.prob_above_with_iv(strike, wa_iv, spot=spot_a, kalshi_close_iso=close)
            db_bid_iv_w = wb_iv
            db_ask_iv_w = wa_iv

        self.db.insert_fill(
            ticker=ticker, action=action, side=side,
            count=count, price=price, strike=strike,
            event_ticker=event_ticker,
            spot_bid=spot_b, spot_ask=spot_a,
            theo_bid=theo_bid, theo_ask=theo_ask,
            kalshi_yes_bid=bk.get("yes_bid", 0),
            kalshi_yes_ask=bk.get("yes_ask", 0),
            deribit_bid_iv=deribit_bid_iv,
            deribit_ask_iv=deribit_ask_iv,
            theo_bid_weekly=theo_bid_w, theo_ask_weekly=theo_ask_w,
            deribit_bid_iv_weekly=db_bid_iv_w, deribit_ask_iv_weekly=db_ask_iv_w,
        )
        print(f"[Recorder] FILL: {ticker} {action} {side} x{count} @ ${price:.2f} "
              f"(theo={theo_bid:.3f}/{theo_ask:.3f} wk={theo_bid_w:.3f}/{theo_ask_w:.3f})")

    def _on_deribit_update(self):
        """Deribit data refreshed — theos will update on next snapshot."""
        pass

    def _shutdown(self):
        print("\n[Recorder] Shutting down...")
        self.running = False
        if self.price_feed:
            self.price_feed.stop()
        if self.ws_feed:
            self.ws_feed.stop()
        if self.weekly_deribit_feed:
            self.weekly_deribit_feed.stop()
        for exp_str, feed in self.deribit_feeds.items():
            feed.stop()
        self.deribit_feeds.clear()
        self.pricers.clear()
        if self.session_id:
            self.db.end_session(self.session_id)
        self.db.close()
        print("[Recorder] Done")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kalshi market recorder")
    parser.add_argument("--export", metavar="DATE",
                        help="Export a day to parquet (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.export:
        db = RecorderDB()
        db.export_parquet("fills", args.export)
        db.export_parquet("market_snapshots", args.export)
        db.close()
        return

    recorder = Recorder()
    recorder.start()


if __name__ == "__main__":
    main()
