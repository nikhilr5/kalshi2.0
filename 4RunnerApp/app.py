"""
Kalshi Bracket Trader — PyQt6 Desktop App

Features:
    - Live YES bid/ask via Kalshi websocket
    - Live BTC/USD via Coinbase websocket
    - Portfolio balance display
    - Sell YES orders at bid + edge
    - Place All Orders button — sells YES on every contract without a resting order
    - Balance check before placing orders
    - Auto-cancel resting orders X hours before expiration
    - Email notification on fills
    - Graceful Ctrl+C shutdown
"""

import sys
import json
import math
import signal
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox,
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QIcon

from kalshi_api import KalshiAPI
from order_manager import OrderManager
from market_discovery import discover_weekly_events, discover_events_for_series, parse_strike
from ws_feed import KalshiWsFeed
from btc_price_feed import CryptoPriceFeed
from vol_smile import VolSmile


# =============================================================================
# Background Workers
# =============================================================================

class DiscoverWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api, series_ticker):
        super().__init__()
        self.api = api
        self.series_ticker = series_ticker

    def run(self):
        try:
            events = discover_events_for_series(self.api, self.series_ticker)
            self.finished.emit(events)
        except Exception as e:
            self.error.emit(str(e))


class OrderWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        try:
            result = self.func(*self.args)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class PlaceAllWorker(QThread):
    """Places orders on all contracts without resting orders. Runs in background."""
    finished = pyqtSignal(object)  # emits {"placed": N, "skipped": N, "failed": N, "errors": [...]}
    progress = pyqtSignal(str)     # emits status updates during placement

    def __init__(self, order_mgr, edge):
        super().__init__()
        self.order_mgr = order_mgr
        self.edge = edge

    def run(self):
        placed = 0
        skipped = 0
        failed = 0
        errors = []

        contracts = list(self.order_mgr.contracts.values())
        total = len(contracts)

        for i, state in enumerate(contracts):
            # Skip if already has a resting order
            if state.order_id:
                skipped += 1
                continue

            # Skip if no bid and no ask (empty market)
            if state.best_bid <= 0 and state.best_ask <= 0:
                skipped += 1
                continue

            self.progress.emit(f"Placing {i+1}/{total}: {state.yes_sub_title or state.ticker}")

            result = self.order_mgr.sell_yes(state.ticker, self.edge)

            if isinstance(result, dict) and "error" in result:
                failed += 1
                errors.append(f"{state.ticker}: {result['error']}")
            else:
                placed += 1

        self.finished.emit({
            "placed": placed,
            "skipped": skipped,
            "failed": failed,
            "errors": errors,
        })


class PortfolioWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, order_mgr):
        super().__init__()
        self.order_mgr = order_mgr

    def run(self):
        self.order_mgr.refresh_orders()
        self.order_mgr.refresh_positions()
        self.order_mgr.refresh_balance()
        self.finished.emit()


# =============================================================================
# Crypto Series — loaded from series.json
# =============================================================================

_SERIES_FILE = Path(__file__).parent / "series.json"
_CONFIG_FILE = Path(__file__).parent / "config.json"

_CONFIG_DEFAULTS = {
    "edge": "0.14",
    "size": "50",
    "auto_cancel_hrs": "2",
    "vol_pct": "50",
    "vol_mode": 0,
    "series_index": 0,
}

def _load_series() -> list[dict]:
    try:
        with open(_SERIES_FILE) as f:
            return json.load(f)
    except Exception:
        return [{"ticker": "KXBTC", "name": "Bitcoin", "coinbase_product": "BTC-USD"}]

def _load_config() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            saved = json.load(f)
        # Merge with defaults so new keys are always present
        return {**_CONFIG_DEFAULTS, **saved}
    except Exception:
        return dict(_CONFIG_DEFAULTS)

def _save_config(cfg: dict):
    try:
        with open(_CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

CRYPTO_SERIES = _load_series()

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _bracket_theo(spot: float, k_low: float, k_high: float | None,
                  T: float, sigma: float) -> float:
    """Theoretical probability that spot lands in [k_low, k_high] at expiry.

    Uses Black-Scholes log-normal model with r=0.
    P(K_low < S_T < K_high) = N(-d2(K_high)) - N(-d2(K_low))
    where d2(K) = [ln(S/K) - sigma^2*T/2] / (sigma*sqrt(T))
    """
    if T <= 0 or sigma <= 0 or spot <= 0:
        return 0.0

    sqrt_t = sigma * math.sqrt(T)
    drift = -0.5 * sigma * sigma * T  # (r - sigma^2/2)*T with r=0

    def p_below(k):
        d2 = (math.log(spot / k) + drift) / sqrt_t
        return _norm_cdf(-d2)

    p_high = p_below(k_high) if (k_high is not None and k_high > 0) else 1.0
    p_low = p_below(k_low) if (k_low is not None and k_low > 0) else 0.0
    return max(p_high - p_low, 0.0)


# =============================================================================
# Main Application
# =============================================================================

class TradingApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kalshi Bracket Trader")
        self.setMinimumSize(1500, 700)

        self._config = _load_config()

        self.api = KalshiAPI()
        self.order_mgr = OrderManager(self.api, default_quantity=int(self._config["size"]))
        self.ws_feed = None
        self.btc_feed = None
        self.events = []
        self.current_event = None
        self.btc_price = 0.0

        # Persists across series switches so auto-cancel works for all events
        # {event_ticker: {"close_time": str, "tickers": set[str]}}
        self._tracked_events: dict[str, dict] = {}

        # Vol smile — calibrated from market prices
        self.vol_smile = VolSmile()

        # Persistent worker references
        self._pending_refresh = False
        self._discover_worker = None
        self._portfolio_worker = None
        self._order_worker = None
        self._cancel_worker = None
        self._place_all_worker = None
        self._refresh_discover_worker = None
        self._smile_worker = None

        self._build_ui()
        self._apply_config()

        # Connect series combo after UI build to avoid triggering discovery twice
        self.series_combo.currentIndexChanged.connect(lambda _: self._on_series_changed())

        # Save config whenever inputs change
        self.edge_input.textChanged.connect(self._persist_config)
        self.size_input.textChanged.connect(self._persist_config)
        self.auto_cancel_input.textChanged.connect(self._persist_config)
        self.vol_input.textChanged.connect(self._persist_config)
        self.vol_mode_combo.currentIndexChanged.connect(self._persist_config)
        self.series_combo.currentIndexChanged.connect(self._persist_config)

        # Start price feed for the initial series
        entry = self._current_series_entry()
        product = entry.get("coinbase_product", "BTC-USD")
        self.price_label_title.setText(f"{entry.get('name', 'BTC')}:")
        self.btc_feed = CryptoPriceFeed(self._on_crypto_price, product_id=product)
        self.btc_feed.start()

        self._discover_events()

        # Timers
        self.portfolio_timer = QTimer()
        self.portfolio_timer.timeout.connect(self._refresh_portfolio)
        self.portfolio_timer.start(10000)

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self._update_countdown)
        self.countdown_timer.start(1000)

        self._table_dirty = False
        self.table_timer = QTimer()
        self.table_timer.timeout.connect(self._flush_table)
        self.table_timer.start(200)

        self._btc_dirty = False
        self.btc_timer = QTimer()
        self.btc_timer.timeout.connect(self._flush_btc_price)
        self.btc_timer.start(500)

        self.auto_cancel_timer = QTimer()
        self.auto_cancel_timer.timeout.connect(self._check_auto_cancel)
        self.auto_cancel_timer.start(60000)

        # Auto-refresh: rediscover events every 5 minutes to pick up new markets
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.timeout.connect(self._auto_refresh)
        self.auto_refresh_timer.start(300000)

        # Recalibrate vol smile every 30 seconds (uses live market mid-prices)
        self.smile_timer = QTimer()
        self.smile_timer.timeout.connect(self._calibrate_smile)
        self.smile_timer.start(30000)

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # --- Top row: Series, Event, Edge, Size, Auto-cancel ---
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Series:"))
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(160)
        for entry in CRYPTO_SERIES:
            self.series_combo.addItem(
                f"{entry['ticker']} \u2014 {entry['name']}", entry
            )
        self.series_combo.setCurrentIndex(0)
        row1.addWidget(self.series_combo)

        row1.addSpacing(15)
        row1.addWidget(QLabel("Event:"))
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(350)
        self.event_combo.currentIndexChanged.connect(self._on_event_changed)
        row1.addWidget(self.event_combo)

        row1.addSpacing(15)
        row1.addWidget(QLabel("Edge:"))
        self.edge_input = QLineEdit("0.14")
        self.edge_input.setMaximumWidth(70)
        self.edge_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edge_input.textChanged.connect(self._mark_dirty)
        row1.addWidget(self.edge_input)

        row1.addSpacing(15)
        row1.addWidget(QLabel("Size:"))
        self.size_input = QLineEdit("50")
        self.size_input.setMaximumWidth(50)
        self.size_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_input.textChanged.connect(self._on_size_changed)
        row1.addWidget(self.size_input)

        row1.addSpacing(15)
        row1.addWidget(QLabel("Auto-cancel (hrs):"))
        self.auto_cancel_input = QLineEdit("2")
        self.auto_cancel_input.setMaximumWidth(40)
        self.auto_cancel_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row1.addWidget(self.auto_cancel_input)

        row1.addSpacing(15)
        row1.addWidget(QLabel("Vol:"))
        self.vol_mode_combo = QComboBox()
        self.vol_mode_combo.addItem("Manual")
        self.vol_mode_combo.addItem("Smile")
        self.vol_mode_combo.setMaximumWidth(80)
        self.vol_mode_combo.currentIndexChanged.connect(self._on_vol_mode_changed)
        row1.addWidget(self.vol_mode_combo)
        self.vol_input = QLineEdit("50")
        self.vol_input.setMaximumWidth(50)
        self.vol_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vol_input.setPlaceholderText("%")
        self.vol_input.textChanged.connect(self._mark_dirty)
        row1.addWidget(self.vol_input)
        self.smile_label = QLabel("")
        self.smile_label.setStyleSheet("color:#5a6270;font-size:10px;")
        row1.addWidget(self.smile_label)

        row1.addStretch()
        layout.addLayout(row1)

        # --- Second row: Price, Balance, Expires, Action buttons ---
        row2 = QHBoxLayout()

        self.price_label_title = QLabel("BTC:")
        row2.addWidget(self.price_label_title)
        self.price_label = QLabel("--")
        self.price_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.price_label.setStyleSheet("color: #22c55e;")
        self.price_label.setMinimumWidth(110)
        row2.addWidget(self.price_label)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Balance:"))
        self.balance_label = QLabel("--")
        self.balance_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.balance_label.setStyleSheet("color: #f59e0b;")
        self.balance_label.setMinimumWidth(150)
        row2.addWidget(self.balance_label)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Profit:"))
        self.payout_label = QLabel("--")
        self.payout_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.payout_label.setStyleSheet("color: #22c55e;")
        self.payout_label.setMinimumWidth(80)
        row2.addWidget(self.payout_label)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Premium:"))
        self.premium_label = QLabel("--")
        self.premium_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.premium_label.setMinimumWidth(130)
        row2.addWidget(self.premium_label)

        row2.addSpacing(15)
        row2.addWidget(QLabel("Expires:"))
        self.countdown_label = QLabel("--:--:--")
        self.countdown_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.countdown_label.setStyleSheet("color: #f59e0b;")
        row2.addWidget(self.countdown_label)

        row2.addStretch()

        # Place All Orders button
        self.place_all_btn = QPushButton("Place All Orders")
        self.place_all_btn.setStyleSheet(
            "QPushButton{background:#22c55e;color:white;padding:8px 16px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#16a34a;}"
        )
        self.place_all_btn.clicked.connect(self._place_all_orders)
        row2.addWidget(self.place_all_btn)

        row2.addSpacing(8)

        # Cancel All Orders button
        self.cancel_all_btn = QPushButton("Cancel All Orders")
        self.cancel_all_btn.setStyleSheet(
            "QPushButton{background:#ef4444;color:white;padding:8px 16px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#dc2626;}"
        )
        self.cancel_all_btn.clicked.connect(self._cancel_all_orders)
        row2.addWidget(self.cancel_all_btn)

        layout.addLayout(row2)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "Contract", "Yes Bid", "Yes Ask", "Vol", "Theo",
            "Sell YES", "Buy YES", "Position", "Profit",
            "Realized PnL", "Open Qty ($)", "Open Level", "Status"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 250)
        header.resizeSection(1, 110)
        header.resizeSection(2, 110)
        header.resizeSection(3, 55)
        header.resizeSection(4, 55)
        header.resizeSection(5, 150)
        header.resizeSection(6, 150)
        header.resizeSection(7, 200)
        header.resizeSection(8, 70)
        header.resizeSection(9, 150)
        header.resizeSection(10, 90)
        header.resizeSection(11, 80)
        header.resizeSection(12, 70)
        header.setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # --- Status ---
        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("color:#5a6270;font-size:11px;")
        layout.addWidget(self.status_label)

    # =========================================================================
    # BTC Price
    # =========================================================================

    def _on_crypto_price(self, price: float):
        self.btc_price = price
        self._btc_dirty = True

    def _flush_btc_price(self):
        if self._btc_dirty:
            self._btc_dirty = False
            if self.btc_price > 0:
                p = self.btc_price
                if p >= 100:
                    txt = f"${p:,.2f}"
                elif p >= 1:
                    txt = f"${p:,.4f}"
                else:
                    txt = f"${p:,.6f}"
                self.price_label.setText(txt)

    # =========================================================================
    # Series Selection
    # =========================================================================

    def _current_series_entry(self) -> dict:
        """Return the full series dict from the selected dropdown item."""
        return self.series_combo.currentData() or {}

    def _current_series(self) -> str:
        """Return the series ticker from the selected dropdown item."""
        return self._current_series_entry().get("ticker", "")

    def _on_series_changed(self):
        """User picked a different series — clear state, switch price feed, rediscover."""
        if self.ws_feed:
            self.ws_feed.stop()
            self.ws_feed = None
        self.current_event = None
        self.order_mgr.contracts.clear()
        self.table.setRowCount(0)
        self.event_combo.clear()
        self.events = []

        # Switch the live price feed to the new crypto
        entry = self._current_series_entry()
        product = entry.get("coinbase_product", "BTC-USD")
        name = entry.get("name", "BTC")

        self.price_label_title.setText(f"{name}:")
        self.price_label.setText("--")
        self.btc_price = 0.0

        if self.btc_feed:
            self.btc_feed.stop()
        self.btc_feed = CryptoPriceFeed(self._on_crypto_price, product_id=product)
        self.btc_feed.start()

        self._discover_events()

    # =========================================================================
    # Event Discovery
    # =========================================================================

    def _discover_events(self):
        series = self._current_series()
        if not series:
            return
        self.status_label.setText(f"Discovering {series} events...")
        self._discover_worker = DiscoverWorker(self.api, series)
        self._discover_worker.finished.connect(self._on_events_discovered)
        self._discover_worker.error.connect(
            lambda e: self.status_label.setText(f"Discovery error: {e}")
        )
        self._discover_worker.start()

    def _on_events_discovered(self, events):
        self.events = events
        self.event_combo.clear()
        for ev in events:
            self.event_combo.addItem(
                f"{ev['event_ticker']} ({ev['num_brackets']} brackets)"
            )
        self.status_label.setText(f"Found {len(events)} events")

    # =========================================================================
    # Auto-Refresh (pick up new events / markets every 5 min)
    # =========================================================================

    def _auto_refresh(self):
        """Re-discover events for the current series to pick up new markets."""
        if self._refresh_discover_worker and self._refresh_discover_worker.isRunning():
            return
        series = self._current_series()
        if not series:
            return
        self._refresh_discover_worker = DiscoverWorker(self.api, series)
        self._refresh_discover_worker.finished.connect(self._on_auto_refresh_done)
        self._refresh_discover_worker.error.connect(
            lambda e: self.status_label.setText(f"Auto-refresh error: {e}")
        )
        self._refresh_discover_worker.start()

    def _on_auto_refresh_done(self, fresh_events):
        old_event_tickers = {ev["event_ticker"] for ev in self.events}

        # Add completely new events to the dropdown
        added = []
        for ev in fresh_events:
            if ev["event_ticker"] not in old_event_tickers:
                added.append(ev)
                self.events.append(ev)
                self.event_combo.addItem(
                    f"{ev['event_ticker']} ({ev['num_brackets']} brackets)"
                )

        if added:
            self.status_label.setText(
                f"Auto-refresh: {len(added)} new event(s) found"
            )

        # If we have a current event, check for new markets within it
        if self.current_event:
            ct = self.current_event["event_ticker"]
            for ev in fresh_events:
                if ev["event_ticker"] == ct:
                    existing = set(self.order_mgr.contracts.keys())
                    new_markets = [
                        m for m in ev["markets"]
                        if m["ticker"] not in existing
                    ]
                    if new_markets:
                        for m in new_markets:
                            self.order_mgr.add_contract(
                                m["ticker"], m.get("yes_sub_title", "")
                            )
                        self.current_event["markets"].extend(new_markets)
                        self.current_event["markets"].sort(
                            key=lambda m: parse_strike(m["ticker"])
                        )
                        self._rebuild_table()
                        if self.ws_feed:
                            self.ws_feed.subscribe_tickers(
                                [m["ticker"] for m in new_markets]
                            )
                        self.status_label.setText(
                            f"Auto-refresh: {len(new_markets)} new market(s) added"
                        )
                    break

    # =========================================================================
    # Event Selection
    # =========================================================================

    def _on_event_changed(self, index):
        if index < 0 or index >= len(self.events):
            return
        if self.ws_feed:
            self.ws_feed.stop()
            self.ws_feed = None

        self.current_event = self.events[index]
        markets = self.current_event["markets"]
        markets.sort(key=lambda m: parse_strike(m["ticker"]))

        self.order_mgr.contracts.clear()
        for m in markets:
            self.order_mgr.add_contract(m["ticker"], m.get("yes_sub_title", ""))

        # Track for auto-cancel (persists across series switches)
        self._tracked_events[self.current_event["event_ticker"]] = {
            "close_time": self.current_event.get("close_time", ""),
            "tickers": {m["ticker"] for m in markets},
        }

        self._rebuild_table()

        tickers = [m["ticker"] for m in markets]
        self.ws_feed = KalshiWsFeed(self.api, self._on_ws_update)
        self.ws_feed.start(tickers)
        self.status_label.setText(f"WS connected: {len(tickers)} tickers")
        self._refresh_portfolio()

    def _on_size_changed(self, text):
        try:
            size = int(text)
            if size > 0:
                self.order_mgr.default_quantity = size
        except ValueError:
            pass

    # =========================================================================
    # Websocket
    # =========================================================================

    def _on_ws_update(self, ticker, yes_bid, yes_ask, bid_size=0, ask_size=0):
        self.order_mgr.update_book(ticker, yes_bid, yes_ask, bid_size, ask_size)
        self._table_dirty = True

    def _mark_dirty(self):
        self._table_dirty = True

    def _flush_table(self):
        if self._table_dirty:
            self._table_dirty = False
            self._update_table()

    # =========================================================================
    # Portfolio
    # =========================================================================

    def _refresh_portfolio(self):
        if self._portfolio_worker and self._portfolio_worker.isRunning():
            self._pending_refresh = True
            return
        self._pending_refresh = False
        self._portfolio_worker = PortfolioWorker(self.order_mgr)
        self._portfolio_worker.finished.connect(self._on_portfolio_done)
        self._portfolio_worker.start()

    def _on_portfolio_done(self):
        self._table_dirty = True
        bal = self.order_mgr.balance_dollars
        self.balance_label.setText(f"${bal:,.2f}")

        # Total profit across all positions (if all win)
        total_profit = sum(
            abs(s.position_qty) - s.position_exposure
            for s in self.order_mgr.contracts.values()
            if s.position_qty != 0
        )
        if any(s.position_qty != 0 for s in self.order_mgr.contracts.values()):
            self.payout_label.setText(f"${total_profit:,.2f}")
        else:
            self.payout_label.setText("--")

        # Total premium: sum of avg YES sell prices across short positions
        # If this sum > $1.00, you profit no matter which bracket settles true
        total_premium = 0.0
        has_short = False
        for s in self.order_mgr.contracts.values():
            if s.position_qty < 0 and s.position_exposure > 0:
                has_short = True
                avg = s.position_exposure / abs(s.position_qty)
                yes_avg = 1.0 - avg  # YES price you sold at
                total_premium += yes_avg
        if has_short:
            self.premium_label.setText(f"${total_premium:.2f} / $1.00")
            if total_premium >= 1.0:
                self.premium_label.setStyleSheet("color: #22c55e;")  # green — safe
            else:
                self.premium_label.setStyleSheet("color: #ef4444;")  # red — not covered
        else:
            self.premium_label.setText("--")
            self.premium_label.setStyleSheet("color: #5a6270;")

        if self._pending_refresh:
            self._pending_refresh = False
            self._refresh_portfolio()

    # =========================================================================
    # Auto-Cancel
    # =========================================================================

    def _check_auto_cancel(self):
        """Check ALL tracked events (not just the current one) for auto-cancel."""
        if self._cancel_worker and self._cancel_worker.isRunning():
            return
        try:
            hours = float(self.auto_cancel_input.text())
        except ValueError:
            return

        tickers_to_cancel = set()
        events_done = []

        for event_ticker, info in self._tracked_events.items():
            close_str = info.get("close_time", "")
            if not close_str:
                continue
            try:
                close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                now = datetime.now(close.tzinfo)
                remaining_hours = (close - now).total_seconds() / 3600.0
                if remaining_hours <= 0:
                    events_done.append(event_ticker)
                elif remaining_hours <= hours:
                    tickers_to_cancel.update(info["tickers"])
                    events_done.append(event_ticker)
            except Exception:
                continue

        # Clean up fully expired events that aren't being cancelled
        for et in events_done:
            if et not in tickers_to_cancel:
                self._tracked_events.pop(et, None)

        if not tickers_to_cancel:
            return

        self.status_label.setText(
            f"Auto-cancelling orders for {len(events_done)} event(s)..."
        )
        self._cancel_worker = OrderWorker(
            self._do_auto_cancel, tickers_to_cancel, events_done
        )
        self._cancel_worker.finished.connect(self._on_auto_cancel_done)
        self._cancel_worker.error.connect(
            lambda e: self.status_label.setText(f"Auto-cancel error: {e}")
        )
        self._cancel_worker.start()

    def _do_auto_cancel(self, market_tickers: set, events_done: list) -> dict:
        """Cancel resting orders matching the given tickers. Runs in background."""
        orders = self.api.get_orders(status="resting")
        cancelled = 0
        for o in orders:
            if o.get("ticker") in market_tickers:
                try:
                    self.api.cancel_order(o["order_id"])
                    cancelled += 1
                except Exception:
                    pass
        # Clear state for any currently-tracked contracts
        for t in market_tickers:
            state = self.order_mgr.contracts.get(t)
            if state:
                state.order_id = ""
                state.open_level = 0.0
                state.open_quantity_dollars = 0.0
                state.prev_remaining = 0.0
        return {"cancelled": cancelled, "events_done": events_done}

    def _on_auto_cancel_done(self, result):
        cancelled = result.get("cancelled", 0)
        for et in result.get("events_done", []):
            self._tracked_events.pop(et, None)
        self.status_label.setText(f"Auto-cancelled: {cancelled} orders")
        self._table_dirty = True

    # =========================================================================
    # Table
    # =========================================================================

    def _rebuild_table(self):
        contracts = list(self.order_mgr.contracts.values())
        self.table.setRowCount(len(contracts))
        for row, state in enumerate(contracts):
            self.table.setItem(row, 0, QTableWidgetItem(
                state.yes_sub_title or state.ticker
            ))
            for col in (1, 2, 3, 4):
                self.table.setItem(row, col, QTableWidgetItem("--"))
            # Sell YES button
            sell_btn = QPushButton("--")
            sell_btn.setStyleSheet(
                "QPushButton{background:#1e2736;color:#ef4444;padding:4px 6px;"
                "border:1px solid #ef4444;border-radius:3px;"
                "font-weight:bold;font-size:10px;}"
                "QPushButton:hover{background:#ef4444;color:white;}"
            )
            sell_btn.clicked.connect(lambda _, t=state.ticker: self._place_order(t))
            self.table.setCellWidget(row, 5, sell_btn)
            # Buy YES (IOC) button
            buy_btn = QPushButton("--")
            buy_btn.setStyleSheet(
                "QPushButton{background:#1e2736;color:#22c55e;padding:4px 6px;"
                "border:1px solid #22c55e;border-radius:3px;"
                "font-weight:bold;font-size:10px;}"
                "QPushButton:hover{background:#22c55e;color:white;}"
            )
            buy_btn.clicked.connect(lambda _, t=state.ticker: self._buy_yes_ioc(t))
            self.table.setCellWidget(row, 6, buy_btn)
            for col in range(7, 13):
                self.table.setItem(row, col, QTableWidgetItem("--"))

    def _update_table(self):
        edge = self._get_edge()
        flat_sigma = self._get_vol()
        use_smile = self._use_smile()
        T = self._get_time_to_expiry()
        spot = self.btc_price
        contracts = list(self.order_mgr.contracts.values())

        # Pre-compute sorted strikes for bracket bounds
        strikes = [parse_strike(c.ticker) for c in contracts]

        for row, state in enumerate(contracts):
            if row >= self.table.rowCount():
                break

            # Yes Bid
            item = self.table.item(row, 1)
            if item:
                if state.best_bid > 0:
                    item.setText(f"${state.best_bid:.2f} ({state.best_bid_size})")
                    item.setForeground(QColor("#22c55e"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Yes Ask
            item = self.table.item(row, 2)
            if item:
                if state.best_ask > 0:
                    item.setText(f"${state.best_ask:.2f} ({state.best_ask_size})")
                    item.setForeground(QColor("#ef4444"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Vol & Theo
            k_low = strikes[row]
            k_high = strikes[row + 1] if row + 1 < len(strikes) else None
            if use_smile and k_low > 0:
                mid_k = (k_low + k_high) / 2.0 if (k_high and k_high > 0) else k_low
                sigma = self.vol_smile.vol_at(mid_k)
            else:
                sigma = flat_sigma

            # Col 3: Vol
            item = self.table.item(row, 3)
            if item:
                if sigma > 0 and spot > 0 and T > 0:
                    item.setText(f"{sigma:.0%}")
                else:
                    item.setText("--")

            # Col 4: Theo
            item = self.table.item(row, 4)
            if item:
                theo = _bracket_theo(spot, k_low, k_high, T, sigma)
                if theo > 0 and spot > 0 and T > 0:
                    item.setText(f"${theo:.2f}")
                else:
                    item.setText("--")

            # Sell YES button
            btn = self.table.cellWidget(row, 5)
            if btn and isinstance(btn, QPushButton):
                if state.best_bid > 0:
                    sell_price = round(state.best_bid + edge, 2)
                else:
                    sell_price = round(edge, 2)
                btn.setText(f"Sell YES @ ${sell_price:.2f}")

            # Buy YES (IOC) button
            btn = self.table.cellWidget(row, 6)
            if btn and isinstance(btn, QPushButton):
                if state.best_ask > 0:
                    btn.setText(f"Buy YES @ ${state.best_ask:.2f}")
                else:
                    btn.setText("--")

            # Position
            item = self.table.item(row, 7)
            if item:
                if state.position_qty != 0:
                    avg = state.position_exposure / abs(state.position_qty)
                    yes_avg = avg if state.position_qty > 0 else 1.0 - avg
                    item.setText(
                        f"${state.position_exposure:.2f} ({state.position_qty:+d}) "
                        f"@ ${yes_avg:.2f}"
                    )
                    item.setForeground(
                        QColor("#22c55e") if state.position_qty > 0
                        else QColor("#ef4444")
                    )
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Profit if this position wins at settlement
            item = self.table.item(row, 8)
            if item:
                if state.position_qty != 0:
                    profit = abs(state.position_qty) - state.position_exposure
                    item.setText(f"${profit:.2f}")
                    item.setForeground(QColor("#22c55e") if profit > 0 else QColor("#ef4444"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Realized PnL if you close at the contra side
            item = self.table.item(row, 9)
            if item:
                if state.position_qty != 0 and state.position_exposure > 0:
                    avg = state.position_exposure / abs(state.position_qty)
                    qty = abs(state.position_qty)
                    if state.position_qty < 0:
                        yes_avg = 1.0 - avg
                        if state.best_ask > 0:
                            pnl_per = yes_avg - state.best_ask
                            pnl_total = pnl_per * qty
                            item.setText(f"${pnl_per:+.2f}/sh (${pnl_total:+.2f})")
                            item.setForeground(QColor("#22c55e") if pnl_total >= 0 else QColor("#ef4444"))
                        else:
                            item.setText("no ask")
                            item.setForeground(QColor("#5a6270"))
                    else:
                        yes_avg = avg
                        if state.best_bid > 0:
                            pnl_per = state.best_bid - yes_avg
                            pnl_total = pnl_per * qty
                            item.setText(f"${pnl_per:+.2f}/sh (${pnl_total:+.2f})")
                            item.setForeground(QColor("#22c55e") if pnl_total >= 0 else QColor("#ef4444"))
                        else:
                            item.setText("no bid")
                            item.setForeground(QColor("#5a6270"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Open Qty
            item = self.table.item(row, 10)
            if item:
                item.setText(f"${state.open_quantity_dollars:.2f}" if state.open_quantity_dollars > 0 else "--")

            # Open Level
            item = self.table.item(row, 11)
            if item:
                item.setText(f"${state.open_level:.2f}" if state.open_level > 0 else "--")

            # Status
            item = self.table.item(row, 12)
            if item:
                if state.order_id:
                    item.setText("RESTING")
                    item.setForeground(QColor("#f59e0b"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

    # =========================================================================
    # Countdown
    # =========================================================================

    def _update_countdown(self):
        if not self.current_event:
            self.countdown_label.setText("--:--:--")
            return
        close_str = self.current_event.get("close_time", "")
        if not close_str:
            self.countdown_label.setText("--:--:--")
            return
        try:
            close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            now = datetime.now(close.tzinfo)
            remaining = close - now
            if remaining.total_seconds() <= 0:
                self.countdown_label.setText("EXPIRED")
                self.countdown_label.setStyleSheet("color:#ef4444;")
                return
            d = remaining.days
            h, rem = divmod(remaining.seconds, 3600)
            m, s = divmod(rem, 60)
            if d > 0:
                self.countdown_label.setText(f"{d}d {h:02d}:{m:02d}:{s:02d}")
            else:
                self.countdown_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
            secs = remaining.total_seconds()
            if secs < 3600:
                self.countdown_label.setStyleSheet("color:#ef4444;")
            elif secs < 86400:
                self.countdown_label.setStyleSheet("color:#f59e0b;")
            else:
                self.countdown_label.setStyleSheet("color:#22c55e;")
        except Exception:
            self.countdown_label.setText("--:--:--")

    # =========================================================================
    # Place Single Order
    # =========================================================================

    def _place_order(self, ticker: str):
        edge = self._get_edge()
        state = self.order_mgr.contracts.get(ticker)
        if not state:
            QMessageBox.warning(self, "Error", f"Contract not found: {ticker}")
            return

        if state.best_bid > 0:
            sell_price = round(state.best_bid + edge, 2)
        else:
            sell_price = round(edge, 2)

        if sell_price <= 0 or sell_price >= 1.0:
            QMessageBox.warning(self, "Error", f"Invalid price: ${sell_price:.2f}")
            return

        qty = self.order_mgr.default_quantity
        available = self.order_mgr.balance_dollars
        max_loss_per = round(1.0 - sell_price, 2)
        collateral = max_loss_per * qty
        profit_if_no = sell_price * qty

        if collateral > available:
            max_qty = int(available / max_loss_per) if max_loss_per > 0 else 0
            QMessageBox.warning(
                self, "Insufficient Balance",
                f"Not enough balance!\n\n"
                f"You want to sell {qty} YES @ ${sell_price:.2f}\n"
                f"Collateral needed: {qty} × ${max_loss_per:.2f} = ${collateral:.2f}\n"
                f"Available balance: ${available:.2f}\n\n"
                f"Max you can afford: {max_qty} contracts"
            )
            return

        reply = QMessageBox.question(
            self, "Confirm Order",
            f"Sell {qty} YES contracts\n"
            f"{state.yes_sub_title or ticker}\n"
            f"\n"
            f"Sell price:           ${sell_price:.2f}\n"
            f"Collateral (at risk): ${collateral:.2f}  ({qty} × ${max_loss_per:.2f})\n"
            f"Profit if NO:         ${profit_if_no:.2f}  ({qty} × ${sell_price:.2f})\n"
            f"\n"
            f"Available balance:    ${available:.2f}\n"
            f"Balance after:        ${available - collateral:.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.status_label.setText(f"Selling YES for {ticker}...")
        self._order_worker = OrderWorker(self.order_mgr.sell_yes, ticker, edge)
        self._order_worker.finished.connect(
            lambda r: self._on_order_done(r, ticker)
        )
        self._order_worker.error.connect(
            lambda e: self.status_label.setText(f"Order error: {e}")
        )
        self._order_worker.start()

    def _on_order_done(self, result, ticker):
        if isinstance(result, dict) and "error" in result:
            QMessageBox.warning(self, "Order Failed", result["error"])
            self.status_label.setText(f"Order failed: {result['error']}")
        else:
            oid = (result.get("order", {}).get("order_id", "?")
                   if isinstance(result, dict) else "?")
            self.status_label.setText(f"Order placed: {oid}")
        QTimer.singleShot(1500, self._refresh_portfolio)

    # =========================================================================
    # Buy YES IOC (close position)
    # =========================================================================

    def _buy_yes_ioc(self, ticker: str):
        state = self.order_mgr.contracts.get(ticker)
        if not state:
            QMessageBox.warning(self, "Error", f"Contract not found: {ticker}")
            return
        if state.best_ask <= 0:
            QMessageBox.warning(self, "Error", "No ask available")
            return

        buy_price = round(state.best_ask, 2)
        qty = self.order_mgr.default_quantity
        cost = buy_price * qty

        reply = QMessageBox.question(
            self, "Confirm Buy YES (IOC)",
            f"Buy {qty} YES contracts (IOC)\n"
            f"{state.yes_sub_title or ticker}\n"
            f"\n"
            f"Price: ${buy_price:.2f}\n"
            f"Cost:  ${cost:.2f}  ({qty} x ${buy_price:.2f})\n"
            f"\n"
            f"This is an immediate-or-cancel order.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.status_label.setText(f"Buying YES for {ticker} (IOC)...")
        self._order_worker = OrderWorker(self.order_mgr.buy_yes_ioc, ticker)
        self._order_worker.finished.connect(
            lambda r: self._on_buy_ioc_done(r, ticker)
        )
        self._order_worker.error.connect(
            lambda e: self.status_label.setText(f"Buy IOC error: {e}")
        )
        self._order_worker.start()

    def _on_buy_ioc_done(self, result, ticker):
        if isinstance(result, dict) and "error" in result:
            QMessageBox.warning(self, "Buy Failed", result["error"])
            self.status_label.setText(f"Buy IOC failed: {result['error']}")
        else:
            oid = (result.get("order", {}).get("order_id", "?")
                   if isinstance(result, dict) else "?")
            self.status_label.setText(f"Buy IOC sent: {oid}")
        # Delay refresh to let Kalshi settle the fill and avoid skipping
        # if a portfolio worker is already running
        QTimer.singleShot(1500, self._refresh_portfolio)

    # =========================================================================
    # Place All Orders
    # =========================================================================

    def _place_all_orders(self):
        """Sell YES on every contract that doesn't already have a resting order."""
        edge = self._get_edge()
        qty = self.order_mgr.default_quantity

        # Count how many orders will be placed
        eligible = []
        for state in self.order_mgr.contracts.values():
            if state.order_id:
                continue  # already has resting order
            if state.best_bid <= 0 and state.best_ask <= 0:
                continue  # empty market
            sell_price = round(state.best_bid + edge, 2) if state.best_bid > 0 else round(edge, 2)
            if 0 < sell_price < 1.0:
                eligible.append((state.yes_sub_title or state.ticker, sell_price))

        if not eligible:
            QMessageBox.information(self, "Place All", "No contracts need orders.\nAll have resting orders or empty books.")
            return

        # Build summary
        total_contracts = len(eligible)
        already_resting = sum(1 for s in self.order_mgr.contracts.values() if s.order_id)

        summary_lines = []
        for name, price in eligible[:10]:
            summary_lines.append(f"  {name} @ ${price:.2f}")
        if len(eligible) > 10:
            summary_lines.append(f"  ... and {len(eligible) - 10} more")

        reply = QMessageBox.question(
            self, "Place All Orders",
            f"Sell YES on {total_contracts} contracts\n"
            f"({already_resting} already have resting orders)\n"
            f"Size: {qty} each | Edge: {edge}\n"
            f"\n"
            f"Orders to place:\n"
            + "\n".join(summary_lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Disable button during placement
        self.place_all_btn.setEnabled(False)
        self.place_all_btn.setText("Placing...")
        self.status_label.setText("Placing all orders...")

        self._place_all_worker = PlaceAllWorker(self.order_mgr, edge)
        self._place_all_worker.progress.connect(
            lambda msg: self.status_label.setText(msg)
        )
        self._place_all_worker.finished.connect(self._on_place_all_done)
        self._place_all_worker.start()

    def _on_place_all_done(self, result):
        """Handle Place All completion."""
        self.place_all_btn.setEnabled(True)
        self.place_all_btn.setText("Place All Orders")

        placed = result.get("placed", 0)
        skipped = result.get("skipped", 0)
        failed = result.get("failed", 0)
        errors = result.get("errors", [])

        self.status_label.setText(
            f"Place All: {placed} placed, {skipped} skipped, {failed} failed"
        )

        # Show errors if any
        if errors:
            error_text = "\n".join(errors[:10])
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more"
            QMessageBox.warning(
                self, "Place All — Errors",
                f"Placed: {placed}\nSkipped: {skipped}\nFailed: {failed}\n\nErrors:\n{error_text}"
            )

        self._refresh_portfolio()

    # =========================================================================
    # Cancel All
    # =========================================================================

    def _cancel_all_orders(self):
        reply = QMessageBox.question(
            self, "Cancel All", "Cancel ALL open orders?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.status_label.setText("Cancelling all orders...")
        self._cancel_worker = OrderWorker(self.order_mgr.cancel_all_orders)
        self._cancel_worker.finished.connect(
            lambda r: self.status_label.setText(f"Cancelled: {r}")
        )
        self._cancel_worker.error.connect(
            lambda e: self.status_label.setText(f"Cancel error: {e}")
        )
        self._cancel_worker.start()

    # =========================================================================
    # Config Persistence
    # =========================================================================

    def _apply_config(self):
        """Apply saved config values to UI inputs."""
        self.edge_input.setText(self._config.get("edge", "0.14"))
        self.size_input.setText(self._config.get("size", "50"))
        self.auto_cancel_input.setText(self._config.get("auto_cancel_hrs", "2"))
        self.vol_input.setText(self._config.get("vol_pct", "50"))
        vol_mode = int(self._config.get("vol_mode", 0))
        if 0 <= vol_mode < self.vol_mode_combo.count():
            self.vol_mode_combo.setCurrentIndex(vol_mode)
        idx = int(self._config.get("series_index", 0))
        if 0 <= idx < self.series_combo.count():
            self.series_combo.setCurrentIndex(idx)

    def _persist_config(self):
        """Save current UI values to config file."""
        self._config["edge"] = self.edge_input.text()
        self._config["size"] = self.size_input.text()
        self._config["auto_cancel_hrs"] = self.auto_cancel_input.text()
        self._config["vol_pct"] = self.vol_input.text()
        self._config["vol_mode"] = self.vol_mode_combo.currentIndex()
        self._config["series_index"] = self.series_combo.currentIndex()
        _save_config(self._config)

    # =========================================================================
    # Vol Smile
    # =========================================================================

    def _on_vol_mode_changed(self, index):
        """Toggle between manual vol input and smile-calibrated vol."""
        if index == 0:
            # Manual mode — enable the text input
            self.vol_input.setEnabled(True)
            self.smile_label.setText("")
        else:
            # Smile mode — disable text input, calibrate immediately
            self.vol_input.setEnabled(False)
            self._calibrate_smile()
        self._table_dirty = True

    def _calibrate_smile(self):
        """Kick off smile calibration — fetches orderbooks via REST then fits."""
        if self.vol_mode_combo.currentIndex() != 1:
            return  # only calibrate in smile mode

        contracts = list(self.order_mgr.contracts.values())
        if not contracts or self.btc_price <= 0:
            self.smile_label.setText("no data")
            return

        T = self._get_time_to_expiry()
        if T <= 0:
            self.smile_label.setText("expired")
            return

        # Don't stack up workers
        if self._smile_worker and self._smile_worker.isRunning():
            return

        tickers = [c.ticker for c in contracts]
        self._smile_worker = OrderWorker(self._fetch_smile_data, tickers)
        self._smile_worker.finished.connect(self._on_smile_data)
        self._smile_worker.error.connect(
            lambda e: self.smile_label.setText(f"err: {e}")
        )
        self._smile_worker.start()

    def _fetch_smile_data(self, tickers: list[str]) -> dict:
        """Fetch orderbooks for all tickers via REST. Runs in background."""
        result = {}
        fetched = 0
        both_sides = 0
        for ticker in tickers:
            try:
                book = self.api.get_orderbook(ticker, depth=1)
                yes_levels = book.get("yes", [])
                no_levels = book.get("no", [])
                # Debug: print first ticker's raw response
                if fetched == 0:
                    print(f"[Smile] Sample book keys: {book.keys()}, yes={yes_levels}, no={no_levels}")
                # Best YES bid = highest yes price
                yes_bid = max((p for p, q in yes_levels), default=0.0)
                # Best YES ask = 1 - highest no price
                best_no_bid = max((p for p, q in no_levels), default=0.0)
                yes_ask = round(1.0 - best_no_bid, 2) if best_no_bid > 0 else 0.0
                result[ticker] = (yes_bid, yes_ask)
                fetched += 1
                if yes_bid > 0 and yes_ask > 0:
                    both_sides += 1
            except Exception as e:
                print(f"[Smile] Error fetching {ticker}: {e}")
                result[ticker] = (0.0, 0.0)
        print(f"[Smile] Fetched {fetched}/{len(tickers)}, both bid+ask: {both_sides}")
        return result

    def _on_smile_data(self, book_data: dict):
        """REST orderbook data arrived — fit the smile."""
        if self.vol_mode_combo.currentIndex() != 1:
            return

        contracts = list(self.order_mgr.contracts.values())
        if not contracts or self.btc_price <= 0:
            return

        T = self._get_time_to_expiry()
        if T <= 0:
            return

        strikes = [parse_strike(c.ticker) for c in contracts]
        mid_prices = []
        for state in contracts:
            # Use REST data, fall back to websocket data
            rest_bid, rest_ask = book_data.get(state.ticker, (0.0, 0.0))
            bid = rest_bid if rest_bid > 0 else state.best_bid
            ask = rest_ask if rest_ask > 0 else state.best_ask

            # Only use brackets where both bid and ask exist
            if bid > 0 and ask > 0:
                mid_prices.append((bid + ask) / 2.0)
            else:
                mid_prices.append(0.0)

        usable = sum(1 for p in mid_prices if p > 0)
        print(f"[Smile] Usable mid-prices: {usable}/{len(mid_prices)}, spot={self.btc_price:.0f}, T={T:.6f}")

        ok = self.vol_smile.calibrate(self.btc_price, strikes, mid_prices, T)
        print(f"[Smile] Calibrated: {ok}, n_points={self.vol_smile.n_points}, raw_ivs={self.vol_smile.raw_ivs[:5]}")
        if ok:
            a, b, c = self.vol_smile.params()
            self.vol_input.setText(f"{a * 100:.1f}")
            self.smile_label.setText(
                f"b={b:+.2f} c={c:+.2f} ({self.vol_smile.n_points}pts)"
            )
        else:
            self.smile_label.setText(f"fit failed ({self.vol_smile.n_points}pts)")
        self._table_dirty = True

    def _use_smile(self) -> bool:
        """Whether to use smile vols instead of manual vol."""
        return self.vol_mode_combo.currentIndex() == 1 and self.vol_smile.calibrated

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_edge(self) -> float:
        try:
            return float(self.edge_input.text())
        except ValueError:
            return 0.14

    def _get_vol(self) -> float:
        """Return annualized vol as a decimal (e.g. 50% → 0.50)."""
        try:
            return float(self.vol_input.text()) / 100.0
        except ValueError:
            return 0.50

    def _get_time_to_expiry(self) -> float:
        """Return time to expiry in years."""
        if not self.current_event:
            return 0.0
        close_str = self.current_event.get("close_time", "")
        if not close_str:
            return 0.0
        try:
            close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            now = datetime.now(close.tzinfo)
            return max((close - now).total_seconds() / (365.25 * 24 * 3600), 0.0)
        except Exception:
            return 0.0

    def shutdown(self):
        print("\nShutting down...")
        if self.ws_feed:
            self.ws_feed.stop()
        if self.btc_feed:
            self.btc_feed.stop()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    # Mac dock name
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = "4Runner Trader"
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("4Runner Trader")

    try:
        app.setWindowIcon(QIcon("icon.png"))
    except Exception:
        pass

    app.setStyleSheet("""
        QMainWindow{background:#0a0e17;}
        QWidget{background:#0a0e17;color:#c8cdd5;
            font-family:-apple-system,BlinkMacSystemFont,sans-serif;}
        QLabel{color:#c8cdd5;font-size:12px;}
        QComboBox{background:#141923;color:#c8cdd5;border:1px solid #1e2736;
            padding:6px 10px;border-radius:4px;font-size:12px;}
        QComboBox::drop-down{border:none;}
        QComboBox QAbstractItemView{background:#141923;color:#c8cdd5;
            selection-background-color:#1e2736;}
        QLineEdit{background:#141923;color:#c8cdd5;border:1px solid #1e2736;
            padding:6px 10px;border-radius:4px;font-size:12px;}
        QTableWidget{background:#0a0e17;alternate-background-color:#0f1520;
            gridline-color:#1e2736;border:1px solid #1e2736;font-size:11px;}
        QTableWidget::item{padding:4px 8px;}
        QHeaderView::section{background:#141923;color:#5a6270;
            border:1px solid #1e2736;padding:6px 8px;
            font-weight:bold;font-size:11px;}
    """)

    window = TradingApp()
    window.show()

    # Ctrl+C handling
    def sigint_handler(*args):
        window.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, sigint_handler)
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()