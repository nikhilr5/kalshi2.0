"""
Standalone theo engine — fits the IV smile per event and publishes theos
over UDP for the recorder to consume.

Tracks every KXBTCD daily 5pm ET event and the nearest Friday weekly,
across every strike within OTM_FILTER_PCT of spot.  For each event:

    1. Subscribe to its tickers' books on the Kalshi WS
    2. On every Coinbase spot tick or book change, compute mid IV per
       strike, fit a quadratic smile across in-window strikes (with IQR
       outlier rejection), evaluate per strike, EWM-smooth, then compute
       theo = N(d2)
    3. UDP-publish each strike's theo to 127.0.0.1:9871

The recorder binds the same port and writes the messages to its events
table.  If the recorder isn't running, the engine still computes and
just drops packets — no coupling.

Usage:
    python theo_engine.py
"""

import json
import math
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

# Local-app modules sit alongside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalshi_api import KalshiAPI
from market_discovery import discover_events_for_series, parse_strike, display_strike
from btc_price_feed import CryptoPriceFeed
from ws_feed import KalshiWsFeed


SERIES = "KXBTCD"
COINBASE_PRODUCT = "BTC-USD"
OTM_FILTER_PCT = 8.0          # only fit / publish within this band
SMILE_OTM_PCT = 0.04          # strikes used for the smile fit (4% of spot)
RISK_FREE_RATE = 0.043
EWM_SPAN = 60                  # smoothing of fitted IV per strike

# Refit policy: refit smile when spot moves >= this fraction since the
# last fit, OR after MAX_REFIT_GAP_SEC (whichever first).  Between fits,
# theo is recomputed from cached smile coeffs which is cheap.
REFIT_SPOT_PCT = 0.0005       # 5 bps
MAX_REFIT_GAP_SEC = 1.0

PUB_HOST = "127.0.0.1"
PUB_PORT = 9871

_norm = NormalDist()


# =============================================================================
# Math
# =============================================================================

def implied_vol(price: float, spot: float, strike: float, T: float,
                r: float = RISK_FREE_RATE) -> float:
    """Solve N(d2) = price for sigma. Returns 0 on edge prices or unsolvable."""
    if price <= 0.01 or price >= 0.99 or spot <= 0 or strike <= 0 or T <= 0:
        return 0.0
    try:
        x = _norm.inv_cdf(price)
        m = math.log(spot / strike) + r * T
        disc = x * x + 2 * m
        if disc < 0:
            return 0.0
        sqrt_disc = math.sqrt(disc)
        u1 = -x + sqrt_disc
        u2 = -x - sqrt_disc
        candidates = [u for u in (u1, u2) if u > 0]
        if not candidates:
            return 0.0
        return min(candidates) / math.sqrt(T)
    except Exception:
        return 0.0


def prob_above(spot: float, strike: float, sigma: float, T: float,
               r: float = RISK_FREE_RATE) -> float:
    """N(d2) — closed-form probability of finishing above strike."""
    if T <= 0 or sigma <= 0:
        return 1.0 if spot > strike else 0.0
    sqrt_T = math.sqrt(T)
    d2 = (math.log(spot / strike) + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return max(min(0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))), 1.0), 0.0)


# =============================================================================
# Per-event smile state
# =============================================================================

class EventState:
    """One per active event_ticker.  Holds tracked tickers, latest book,
    last smile fit coefficients, and per-strike smoothed IV state."""

    def __init__(self, event_ticker: str, close_time: str):
        self.event_ticker = event_ticker
        self.close_time = close_time
        try:
            self.close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        except Exception:
            self.close_dt = None
        # ticker -> {"display_strike", "raw_strike"}
        self.markets: dict[str, dict] = {}
        # ticker -> {"yes_bid", "yes_ask"}
        self.book: dict[str, dict] = {}
        # Smoothed IV state per display_strike (EWM running mean)
        self.smoothed_iv: dict[float, float] = {}
        # Last theo published per ticker (for change-only publish)
        self.last_published: dict[str, float] = {}
        # Last smile fit
        self.last_fit_spot: float = 0.0
        self.last_fit_ts: float = 0.0
        self.coeffs: tuple[float, float, float] | None = None

    def time_to_expiry(self, now_utc: datetime) -> float:
        """Years to expiry, clamped at 0."""
        if self.close_dt is None:
            return 0.0
        seconds = (self.close_dt - now_utc).total_seconds()
        return max(seconds / (365.25 * 24 * 3600), 0.0)


# =============================================================================
# Engine
# =============================================================================

class TheoEngine:

    def __init__(self):
        self.api = KalshiAPI()
        self.running = False

        self.spot = 0.0
        self.spot_bid = 0.0
        self.spot_ask = 0.0

        self.price_feed = None
        self.ws_feed = None

        # event_ticker -> EventState
        self.events: dict[str, EventState] = {}

        # UDP publisher socket (sender; no bind)
        self.pub_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pub_addr = (PUB_HOST, PUB_PORT)

        # Reverse lookup ticker -> event_ticker
        self._ticker_event: dict[str, str] = {}

    # --- Lifecycle ---

    def start(self):
        self.running = True
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        print(f"[Theo] Discovering {SERIES} events...")
        all_events = discover_events_for_series(self.api, SERIES)
        if not all_events:
            print("[Theo] No events found")
            return

        # Filter to upcoming 5pm-ET-close events (daily + Friday weekly).
        now_utc = datetime.now(tz=timezone.utc)
        for ev in all_events:
            close_str = ev.get("close_time", "")
            if not close_str:
                continue
            try:
                close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if close_utc <= now_utc:
                continue
            # Daily/weekly = closes at 21:00 UTC (5pm ET)
            if close_utc.hour != 21:
                continue
            self._add_event(ev)

        if not self.events:
            print("[Theo] No upcoming 5pm-ET events")
            return

        # Start Coinbase
        print(f"[Theo] Starting Coinbase feed for {COINBASE_PRODUCT}...")
        self.price_feed = CryptoPriceFeed(self._on_price, COINBASE_PRODUCT)
        self.price_feed.start()

        # Wait briefly for spot
        for _ in range(100):
            if self.spot > 0:
                break
            time.sleep(0.1)
        if self.spot <= 0:
            print("[Theo] No spot price yet — proceeding anyway")

        # Filter markets within OTM band (across every event) before subscribing
        if self.spot > 0:
            self._filter_to_band()

        # Subscribe to Kalshi WS for all tracked tickers (one connection)
        tickers = self._all_tickers()
        if not tickers:
            print("[Theo] No tickers to subscribe to after OTM filter")
            return
        self.ws_feed = KalshiWsFeed(self.api, on_update=self._on_ws_update)
        self.ws_feed.start(tickers)
        print(f"[Theo] Subscribed to {len(tickers)} tickers across "
              f"{len(self.events)} events")
        print(f"[Theo] Publishing theos to UDP {PUB_HOST}:{PUB_PORT}")

        self._run_loop()

    def _add_event(self, ev: dict):
        et = ev["event_ticker"]
        st = EventState(et, ev.get("close_time", ""))
        for m in ev.get("markets", []):
            ticker = m["ticker"]
            raw = parse_strike(ticker)
            if raw <= 0:
                continue
            disp = display_strike(raw)
            st.markets[ticker] = {"display_strike": disp, "raw_strike": raw}
            self._ticker_event[ticker] = et
        self.events[et] = st
        print(f"[Theo] Tracking {et} ({len(st.markets)} markets, "
              f"close={st.close_time})")

    def _filter_to_band(self):
        """Trim each event's markets to those within OTM_FILTER_PCT of spot."""
        threshold = OTM_FILTER_PCT
        for ev in self.events.values():
            keep = {}
            for ticker, info in ev.markets.items():
                disp = info["display_strike"]
                otm = abs((disp - self.spot) / self.spot * 100)
                if otm <= threshold:
                    keep[ticker] = info
                else:
                    self._ticker_event.pop(ticker, None)
            ev.markets = keep

    def _all_tickers(self) -> list:
        out = []
        for ev in self.events.values():
            out.extend(ev.markets.keys())
        return out

    def _run_loop(self):
        # Light loop — most work happens in WS callbacks.  We just
        # periodically re-discover to catch new events / drop expired.
        last_refresh = 0.0
        while self.running:
            time.sleep(0.5)
            now = time.time()
            if now - last_refresh > 60:
                last_refresh = now
                self._refresh_events()

    def _refresh_events(self):
        """Drop expired events, pick up new ones (e.g. tomorrow's daily)."""
        try:
            all_events = discover_events_for_series(self.api, SERIES)
        except Exception:
            return
        now_utc = datetime.now(tz=timezone.utc)
        # Drop expired
        for et in list(self.events.keys()):
            ev = self.events[et]
            if ev.close_dt and (now_utc - ev.close_dt).total_seconds() > 600:
                print(f"[Theo] Dropping expired event {et}")
                for ticker in list(ev.markets.keys()):
                    self._ticker_event.pop(ticker, None)
                del self.events[et]
        # Add new
        added_tickers = []
        for ev_dict in all_events:
            et = ev_dict["event_ticker"]
            close_str = ev_dict.get("close_time", "")
            if not close_str or et in self.events:
                continue
            try:
                close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if close_utc <= now_utc or close_utc.hour != 21:
                continue
            self._add_event(ev_dict)
            # Filter the new event's markets to the OTM band
            if self.spot > 0:
                ev = self.events[et]
                keep = {}
                for t, info in ev.markets.items():
                    disp = info["display_strike"]
                    otm = abs((disp - self.spot) / self.spot * 100)
                    if otm <= OTM_FILTER_PCT:
                        keep[t] = info
                        added_tickers.append(t)
                    else:
                        self._ticker_event.pop(t, None)
                ev.markets = keep
        if added_tickers and self.ws_feed:
            self.ws_feed.subscribe_tickers(added_tickers)

    # --- WS callbacks ---

    def _on_price(self, price: float, bid: float = 0.0, ask: float = 0.0):
        self.spot = price
        if bid > 0:
            self.spot_bid = bid
        if ask > 0:
            self.spot_ask = ask
        # Spot moved — recompute every active event's theos
        self._recompute_all()

    def _on_ws_update(self, ticker: str, yes_bid: float, yes_ask: float,
                      bid_size: int = 0, ask_size: int = 0):
        et = self._ticker_event.get(ticker)
        if not et:
            return
        ev = self.events.get(et)
        if not ev:
            return
        ev.book[ticker] = {"yes_bid": yes_bid, "yes_ask": yes_ask}
        # Book moved — recompute just this event's theos
        self._recompute_event(ev)

    # --- Theo computation ---

    def _recompute_all(self):
        for ev in self.events.values():
            self._recompute_event(ev)

    def _recompute_event(self, ev: EventState):
        if self.spot <= 0:
            return
        now_utc = datetime.now(tz=timezone.utc)
        T = ev.time_to_expiry(now_utc)
        if T <= 0:
            return

        now_mono = time.monotonic()
        spot = self.spot
        # Refit decision: spot moved enough or stale
        moved = ev.last_fit_spot == 0 or abs(
            spot - ev.last_fit_spot) / spot >= REFIT_SPOT_PCT
        stale = (now_mono - ev.last_fit_ts) >= MAX_REFIT_GAP_SEC
        if moved or stale or ev.coeffs is None:
            self._refit_smile(ev, T, spot)
            ev.last_fit_spot = spot
            ev.last_fit_ts = now_mono

        if ev.coeffs is None:
            return  # not enough data to fit yet

        a, b, c = ev.coeffs
        for ticker, info in ev.markets.items():
            disp = info["display_strike"]
            fitted_iv = a * disp * disp + b * disp + c
            if fitted_iv <= 0:
                continue
            # EWM smooth: y = (1 - alpha) * prev + alpha * x  with span N
            alpha = 2.0 / (EWM_SPAN + 1.0)
            prev = ev.smoothed_iv.get(disp, fitted_iv)
            smoothed = (1 - alpha) * prev + alpha * fitted_iv
            ev.smoothed_iv[disp] = smoothed
            theo = prob_above(spot, disp, smoothed, T)
            self._publish(ev, ticker, disp, theo, smoothed)

    def _refit_smile(self, ev: EventState, T: float, spot: float):
        """Quadratic fit on mid_iv across strikes within SMILE_OTM_PCT of spot."""
        strikes = []
        ivs = []
        for ticker, info in ev.markets.items():
            disp = info["display_strike"]
            if abs(disp - spot) / spot > SMILE_OTM_PCT:
                continue
            bk = ev.book.get(ticker)
            if not bk:
                continue
            bid = bk.get("yes_bid", 0.0)
            ask = bk.get("yes_ask", 0.0)
            if bid <= 0 or ask <= 0 or bid >= ask:
                continue
            mid = (bid + ask) / 2.0
            iv = implied_vol(mid, spot, disp, T)
            if iv <= 0:
                continue
            strikes.append(disp)
            ivs.append(iv)

        if len(strikes) < 3:
            ev.coeffs = None
            return

        ivs_np = np.array(ivs)
        strikes_np = np.array(strikes)
        q1, q3 = np.percentile(ivs_np, [25, 75])
        iqr = q3 - q1
        mask = (ivs_np >= q1 - 1.5 * iqr) & (ivs_np <= q3 + 1.5 * iqr)
        if mask.sum() < 3:
            ev.coeffs = None
            return
        try:
            ev.coeffs = tuple(np.polyfit(strikes_np[mask], ivs_np[mask], 2))
        except Exception:
            ev.coeffs = None

    # --- Publish ---

    def _publish(self, ev: EventState, ticker: str, strike: float,
                 theo: float, smoothed_iv: float):
        last = ev.last_published.get(ticker)
        if last is not None and abs(last - theo) < 1e-5:
            return
        ev.last_published[ticker] = theo
        bk = ev.book.get(ticker, {})
        msg = {
            "ts_us": int(time.time() * 1_000_000),
            "event_type": "theo",
            "event_ticker": ev.event_ticker,
            "ticker": ticker,
            "payload": {
                "theo": theo,
                "smoothed_iv": smoothed_iv,
                "kalshi_bid": bk.get("yes_bid", 0.0),
                "kalshi_ask": bk.get("yes_ask", 0.0),
                "spot": self.spot,
                "strike": strike,
            },
        }
        try:
            self.pub_sock.sendto(
                json.dumps(msg, separators=(",", ":")).encode("utf-8"),
                self.pub_addr,
            )
        except Exception:
            pass

    # --- Shutdown ---

    def _shutdown(self):
        print("\n[Theo] Shutting down...")
        self.running = False
        if self.price_feed:
            self.price_feed.stop()
        if self.ws_feed:
            self.ws_feed.stop()
        try:
            self.pub_sock.close()
        except Exception:
            pass
        print("[Theo] Done")


if __name__ == "__main__":
    engine = TheoEngine()
    engine.start()
