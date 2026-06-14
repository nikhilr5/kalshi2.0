"""Aston — 15-minute up/down market maker.

Single-strike-per-window contract.  GUI shows: strike, time to expiry,
yes bid/ask, theo (N(d2) from realized vol), the vol itself, and an
edge control for each side.  Start/Stop arms the strategy.

Run:
    python app.py
"""

import json
import signal
import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QMenu, QSpinBox, QTableWidget,
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from kalshi_api import KalshiAPI
from feeds.crypto_feed import CryptoPriceFeed
from feeds.ws_feed import KalshiWsFeed
from feeds.market_discovery import (
    SERIES_15M, get_active_market, parse_strike, seconds_to_close,
)
from pricing.realized_vol import RealizedVolEstimator
from pricing.har_rv import HARRVEstimator
from pricing.theo_engine import compute_theo
from strategy2 import Strategy2
from osm import OSM


SETTINGS_PATH = Path(__file__).resolve().parent / "settings" / "aston_settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}


def _save_settings(d: dict):
    try:
        SETTINGS_PATH.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------
# Market discovery runs on a worker so we don't block the GUI on API
# calls.  Emits the active market dict (or None) when done.
# ---------------------------------------------------------------------

class DiscoverWorker(QThread):
    finished_signal = pyqtSignal(object)  # market dict or None

    def __init__(self, api: KalshiAPI, series_ticker: str):
        super().__init__()
        self.api = api
        self.series_ticker = series_ticker

    def run(self):
        try:
            m = get_active_market(self.api, self.series_ticker)
        except Exception as e:
            print(f"[Discover] error: {e}")
            m = None
        self.finished_signal.emit(m)


class HarSeedWorker(QThread):
    """Pulls 25 hours of 1-minute Coinbase candles to seed HAR-RV.
    25h instead of 24h so the buffer fills with one extra minute of
    cushion past the 1,440-minute requirement."""
    finished_signal = pyqtSignal(list)  # list[(unix_minute_idx, close)]

    def __init__(self, product_id: str):
        super().__init__()
        self.product_id = product_id

    def run(self):
        try:
            import requests
            from datetime import datetime, timedelta, timezone
            url = (f"https://api.exchange.coinbase.com/products/"
                   f"{self.product_id}/candles")
            end = datetime.now(timezone.utc)
            start_overall = end - timedelta(hours=25)
            cursor = end
            rows = []
            while cursor > start_overall:
                batch_start = max(cursor - timedelta(minutes=300),
                                  start_overall)
                r = requests.get(url, params={
                    "granularity": 60,
                    "start": batch_start.isoformat(),
                    "end": cursor.isoformat(),
                }, timeout=10)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                rows.extend(batch)
                cursor = batch_start
            now_minute = int(time.time() // 60)
            # Drop the current in-progress minute so live aggregation
            # owns it exclusively — avoids the seed and on_price each
            # writing a bucket for the same minute index.
            # Coinbase row order: [time, low, high, open, close, volume]
            # HAR-RV (Parkinson) wants (minute, high, low).
            candles = [(int(r[0]) // 60, float(r[2]), float(r[1]))
                       for r in rows
                       if (int(r[0]) // 60) < now_minute]
            self.finished_signal.emit(candles)
        except Exception as e:
            print(f"[HARSeed] error: {e}")
            self.finished_signal.emit([])


class BalanceWorker(QThread):
    """Fetches /portfolio/balance.  Kalshi computes both fields
    server-side — no client-side MtM math needed.

    Response shape:
        balance:         cash available, in cents.
        portfolio_value: mark-to-market of open positions, in cents.
    Total portfolio = balance + portfolio_value.
    """
    finished_signal = pyqtSignal(int, int)  # (balance_cents, portfolio_value_cents)

    def __init__(self, api: KalshiAPI):
        super().__init__()
        self.api = api

    def run(self):
        try:
            data = self.api.get_balance()
            print(f"[Balance] raw response: {data}")
            balance = int(data.get("balance", 0) or 0)
            portfolio_value = int(data.get("portfolio_value", 0) or 0)
        except Exception as e:
            print(f"[Balance] error: {e}")
            self.finished_signal.emit(-1, 0)
            return
        self.finished_signal.emit(balance, portfolio_value)


class PositionSeedWorker(QThread):
    """Pulls the current position + fill history for a single ticker
    so the app can resume from existing exposure on restart.  Returns
    (position_count, avg_entry_dollars).  Avg entry is replayed from
    fills using the same accounting the WS handler uses, so it
    correctly reflects the currently-open exposure even after multiple
    open/close cycles within the same market."""
    finished_signal = pyqtSignal(int, float)

    def __init__(self, api: KalshiAPI, ticker: str):
        super().__init__()
        self.api = api
        self.ticker = ticker

    def run(self):
        try:
            positions = self.api.get_positions()
            pos_dict = next(
                (p for p in positions if p.get("ticker") == self.ticker), None)
            net_pos = int(pos_dict.get("position", 0)) if pos_dict else 0
            if net_pos == 0:
                self.finished_signal.emit(0, 0.0)
                return

            fills = self.api.get_fills(ticker=self.ticker)
            # Kalshi returns fills newest-first by default; replay
            # chronologically so weighted-avg math matches WS path.
            fills = sorted(fills, key=lambda f: f.get("created_time", ""))
            pos = 0
            avg = 0.0
            for f in fills:
                action = f.get("action", "")
                side = f.get("side", "yes")
                count = float(f.get("count", f.get("count_fp", 0)) or 0)
                # Prefer dollar field; fall back to cents if necessary.
                yp_dollars = f.get("yes_price_dollars")
                if yp_dollars is not None and yp_dollars != "":
                    price = float(yp_dollars)
                else:
                    yp = float(f.get("yes_price", 0) or 0)
                    price = yp / 100.0 if yp >= 1.0 else yp

                if (side == "yes" and action == "buy") or \
                   (side == "no" and action == "sell"):
                    delta = count
                else:
                    delta = -count
                prev = pos
                new = prev + int(delta)
                yes_price = price if side == "yes" else (1.0 - price)
                if prev == 0:
                    avg = yes_price
                elif (prev > 0 and delta > 0) or (prev < 0 and delta < 0):
                    avg = (avg * abs(prev) + yes_price * abs(delta)) / abs(new)
                elif new == 0:
                    avg = 0.0
                elif abs(delta) > abs(prev):
                    avg = yes_price
                pos = new
            self.finished_signal.emit(pos, avg)
        except Exception as e:
            print(f"[PositionSeed] error: {e}")
            self.finished_signal.emit(0, 0.0)


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------

class AstonApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aston — 15-min MM (strategy v2)")
        self.resize(720, 460)

        self.settings = _load_settings()

        # Core services.  Hand KalshiAPI the recorder's data dir so
        # every create_order / cancel_order call gets logged off-thread
        # to a daily-rotated JSONL.  Recorder tails those files and
        # ingests into the per-day `order_attempts` table.  Zero
        # latency cost on the trading path (queue put_nowait).
        from tools.recorder import DEFAULT_DATA_DIR as _ATTEMPT_LOG_DIR
        self.api = KalshiAPI(attempt_log_dir=_ATTEMPT_LOG_DIR)
        # Process-lifetime exchange-policy layer (write-token budget).
        # Shared across every OSM rebuild so the budget survives the
        # 15-minute market rolls.
        from order_gateway import OrderGateway
        self.gateway = OrderGateway(self.api)
        self.price_feed: CryptoPriceFeed | None = None
        self.ws_feed: KalshiWsFeed | None = None
        self.vol_est = RealizedVolEstimator(
            lookback_minutes=float(self.settings.get("vol_lookback_min", 30.0)),
            sample_seconds=float(self.settings.get("vol_sample_sec", 10.0)),
        )
        # HAR-RV: primary vol forecast.  Fed the same Coinbase ticks as
        # vol_est, but buckets to 1-min closes and applies fitted OLS
        # coefficients to predict the next 15 minutes of RV.  Falls
        # back to vol_est until 24h of history is seeded.
        self.har_est = HARRVEstimator(
            coef_path=Path(__file__).resolve().parent / "settings" / "har_coefficients.json"
        )
        self._har_seed_worker: HarSeedWorker | None = None
        # Fill persistence is delegated to the standalone `recorder.py`
        # process — see its module docstring.  Nothing to wire here.

        # Active market state
        self.market: dict | None = None       # raw market dict from API
        self.strike: float = 0.0
        self.ticker: str = ""
        self.yes_bid: float = 0.0
        self.yes_ask: float = 0.0
        self.bid_size: int = 0
        self.ask_size: int = 0
        self.spot: float = 0.0
        self.strategy: Strategy | Strategy2 | None = None
        # OSM owns all
        # Kalshi order I/O on behalf of Strategy2.
        self.osm: OSM | None = None
        # Cached theo for display (recomputed on every spot tick)
        self._last_theo: float | None = None
        self._last_sigma: float | None = None
        # Theo-recompute cadence tracking.  EMA (alpha=0.2) of the
        # inter-arrival ms in _recompute_and_trade.  Read by the 250ms
        # UI tick to render `theo_rate_label`.
        self._theo_last_ts: float | None = None
        self._theo_dt_ms: float = 0.0
        # Position is tracked locally from WS fills; the strategy mirrors it.
        # `_avg_entry` is the weighted-average YES price of the currently
        # open position (in dollars).  0 when flat.
        self._position: int = 0
        self._avg_entry: float = 0.0

        # Portfolio — both pulled directly from /portfolio/balance.
        # Total portfolio value displayed = balance + portfolio_value.
        # Auto-MM mode — tracks which ticker we've already auto-engaged
        # so manual OFF after auto-ON isn't re-fought within the same
        # 15-min window.  Reset on auto-mode toggle and on ticker change.
        self._auto_mm_done_ticker: str | None = None

        self._balance_cents: int = -1          # cash (Kalshi `balance`)
        self._portfolio_value_cents: int = 0   # position MtM (Kalshi `portfolio_value`)
        self._balance_worker: BalanceWorker | None = None

        self._build_ui()
        self._apply_stylesheet()

        # Kick off discovery for whatever series is selected by default.
        self._on_series_changed(0)

        # Periodic UI refresh — countdown, theo display, etc.
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._refresh_ui)
        self.ui_timer.start(250)

        # Periodic re-discovery — when the current 15-min window closes
        # we need to roll to the next market.
        self.discover_timer = QTimer(self)
        self.discover_timer.timeout.connect(self._rediscover)
        self.discover_timer.start(15_000)

        # Balance — refresh on launch + every 60s
        self.balance_timer = QTimer(self)
        self.balance_timer.timeout.connect(self._refresh_balance)
        self.balance_timer.start(60_000)
        self._refresh_balance()

        # Live orders panel is driven directly from strategy state on
        # the UI tick — no REST polling required, the strategy already
        # owns the truth (resting_*_id, current_*_price, resting_*_count).

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(14)

        # =============================================================
        # HEADER: series + portfolio + MM toggle
        # =============================================================
        header = QHBoxLayout()
        header.setSpacing(16)

        series_box = QVBoxLayout()
        series_box.setSpacing(4)
        series_title = QLabel("SERIES")
        series_title.setObjectName("metricTitle")
        series_box.addWidget(series_title)
        self.series_combo = QComboBox()
        for s in SERIES_15M:
            self.series_combo.addItem(s["name"])
        self.series_combo.currentIndexChanged.connect(self._on_series_changed)
        series_box.addWidget(self.series_combo)
        header.addLayout(series_box)

        # STRATEGY badge — bright, unmissable indicator of which engine
        # is driving orders.  v1 = legacy Strategy, v2 = Strategy2 + OSM.
        strat_box = QVBoxLayout()
        strat_box.setSpacing(4)
        strat_title = QLabel("STRATEGY")
        strat_title.setObjectName("metricTitle")
        strat_box.addWidget(strat_title)
        self.strategy_badge = QLabel("v2")
        self.strategy_badge.setFont(QFont("Menlo", 18, QFont.Weight.Bold))
        badge_color = "#22c55e"
        self.strategy_badge.setStyleSheet(
            f"color:{badge_color};background:transparent;border:1px solid {badge_color};"
            "border-radius:4px;padding:2px 10px;")
        strat_box.addWidget(self.strategy_badge)
        header.addLayout(strat_box)

        # Cash — pulled from /portfolio/balance every 60s.
        cash_box = QVBoxLayout()
        cash_box.setSpacing(4)
        ct = QLabel("CASH")
        ct.setObjectName("metricTitle")
        cash_box.addWidget(ct)
        self.cash_label = QLabel("--")
        self.cash_label.setFont(QFont("Menlo", 18, QFont.Weight.Bold))
        self.cash_label.setStyleSheet(
            "color:#22c55e;background:transparent;border:none;")
        cash_box.addWidget(self.cash_label)
        header.addLayout(cash_box)

        # Portfolio value = cash + position × yes_mid on the active
        # ticker.  Recomputed every UI tick (250ms) so the MtM tracks
        # the live BBO.  When flat, equals cash.
        portfolio_box = QVBoxLayout()
        portfolio_box.setSpacing(4)
        pt = QLabel("PORTFOLIO")
        pt.setObjectName("metricTitle")
        portfolio_box.addWidget(pt)
        self.portfolio_label = QLabel("--")
        self.portfolio_label.setFont(QFont("Menlo", 18, QFont.Weight.Bold))
        self.portfolio_label.setStyleSheet(
            "color:#22c55e;background:transparent;border:none;")
        portfolio_box.addWidget(self.portfolio_label)
        header.addLayout(portfolio_box)

        # Write-token budget — live read of the gateway's bucket mirror.
        # Refreshed on the 250ms UI tick; read-only, thread-safe.
        tokens_box = QVBoxLayout()
        tokens_box.setSpacing(4)
        tt = QLabel("TOKENS")
        tt.setObjectName("metricTitle")
        tokens_box.addWidget(tt)
        self.tokens_label = QLabel("--")
        self.tokens_label.setFont(QFont("Menlo", 18, QFont.Weight.Bold))
        self.tokens_label.setStyleSheet(
            "color:#22c55e;background:transparent;border:none;")
        tokens_box.addWidget(self.tokens_label)
        header.addLayout(tokens_box)

        header.addStretch()

        # Big on/off toggle + dropdown for auto-mode config.  QToolButton
        # in MenuButtonPopup mode splits into two zones: the main face
        # toggles MM, the small right arrow opens the menu.
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("MM  OFF")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setMinimumWidth(180)
        self.toggle_btn.setMinimumHeight(46)
        self.toggle_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.toggle_btn.clicked.connect(self._on_toggle_clicked)
        self._build_mm_menu()
        self._apply_toggle_style(False)
        header.addWidget(self.toggle_btn)

        outer.addLayout(header)

        # =============================================================
        # MAIN PANEL: live market, theo, vol, time/T, orders
        # Lives inside a styled frame so it visually dominates the window
        # and the settings panel below reads as secondary.
        # =============================================================
        main_panel = QFrame()
        main_panel.setObjectName("mainPanel")
        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        # Top row of stat cards.  Time + T live in ONE card now per
        # your spec — countdown on top, annualized T as a decimal below.
        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.spot_card = self._make_metric("SPOT", "--", "#facc15")
        self.strike_card = self._make_metric("STRIKE", "--", "#c8cdd5")
        self.time_card = self._make_dual_metric(
            "TIME LEFT", "--:--", "#22c55e",
            "T (years)", "--", "#94a3b8")
        self.vol_card = self._make_metric("REALIZED VOL", "--", "#facc15")
        # HAR-RV info — visible "ⓘ HAR-RV" label as a hover affordance
        # plus tooltips on BOTH the label and the whole card, so the
        # tooltip fires no matter where the user hovers on the box.
        # Tooltip text is rebuilt every UI tick in _refresh_vol_tooltip.
        self.vol_info_label = QLabel("ⓘ  HAR-RV details")
        self.vol_info_label.setStyleSheet(
            "color:#7c8595;font-size:10px;font-weight:bold;letter-spacing:1px;"
            "background:transparent;border:none;")
        self.vol_info_label.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.vol_info_label.setToolTip("Loading HAR-RV…")
        self.vol_card.setToolTip("Loading HAR-RV…")
        self.vol_card.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.vol_card.layout().addWidget(self.vol_info_label)
        for w in (self.spot_card, self.strike_card,
                  self.time_card, self.vol_card):
            stats.addWidget(w, 1)
        main_layout.addLayout(stats)

        # Live market quotes + theo
        quotes = QHBoxLayout()
        quotes.setSpacing(10)
        self.bid_card = self._make_metric("YES BID", "--", "#22c55e")
        self.ask_card = self._make_metric("YES ASK", "--", "#ef4444")
        self.theo_card = self._make_metric("THEO", "--", "#a78bfa")
        # Small recompute-rate readout under the theo number.  Driven by
        # the EMA of inter-arrival time in _recompute_and_trade — covers
        # both Coinbase ticker WS ticks and Kalshi book updates, since
        # both call _recompute_and_trade.
        self.theo_rate_label = QLabel("-- ms")
        self.theo_rate_label.setStyleSheet(
            "color:#5a6270;font-size:10px;background:transparent;border:none;")
        self.theo_card.layout().addWidget(self.theo_rate_label)
        self.position_card = self._make_metric("POSITION", "0", "#c8cdd5")
        for w in (self.bid_card, self.ask_card,
                  self.theo_card, self.position_card):
            quotes.addWidget(w, 1)
        main_layout.addLayout(quotes)

        # Live orders panel — small heading + table of resting orders
        # filtered to the current market.
        orders_title = QLabel("LIVE ORDERS")
        orders_title.setObjectName("metricTitle")
        main_layout.addWidget(orders_title)

        # Two fixed rows — one for our buy, one for our sell.  Read from
        # `self.strategy.current_*_price` directly each UI tick (250ms),
        # so price/size update synchronously with quote placement.  No
        # REST polling.
        self.orders_table = QTableWidget()
        self.orders_table.setColumnCount(4)
        self.orders_table.setHorizontalHeaderLabels(["Side", "Price", "Size", "Age"])
        self.orders_table.setRowCount(2)
        self.orders_table.verticalHeader().setVisible(False)
        self.orders_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.orders_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        oh = self.orders_table.horizontalHeader()
        oh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for i, w in enumerate([90, 110, 90, 80]):
            self.orders_table.setColumnWidth(i, w)
        self.orders_table.setMaximumHeight(96)
        self.orders_table.setMinimumHeight(80)
        main_layout.addWidget(self.orders_table)

        outer.addWidget(main_panel)

        # =============================================================
        # SETTINGS PANEL: collapsible, smaller, tucked under main panel
        # =============================================================
        settings_header = QHBoxLayout()
        self.settings_toggle = QToolButton()
        self.settings_toggle.setText("▶  Settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setChecked(False)
        self.settings_toggle.setStyleSheet("""
            QToolButton{background:transparent;color:#5a6270;
                        font-size:11px;font-weight:bold;letter-spacing:1px;
                        border:none;padding:2px 0;}
            QToolButton:hover{color:#c8cdd5;}
        """)
        self.settings_toggle.clicked.connect(self._toggle_settings_panel)
        settings_header.addWidget(self.settings_toggle)
        settings_header.addStretch()
        # Ticker + status live to the right of the settings toggle so
        # they're always visible without a dedicated row.
        self.ticker_label = QLabel("")
        self.ticker_label.setStyleSheet("color:#5a6270;font-size:11px;")
        settings_header.addWidget(self.ticker_label)
        outer.addLayout(settings_header)

        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("settingsPanel")
        sp = QHBoxLayout(self.settings_panel)
        sp.setContentsMargins(12, 10, 12, 10)
        sp.setSpacing(10)

        # Edge inputs are now in CENTS (display + storage).  Old saved
        # values were in dollars (< 1.0); migrate by ×100 on load so
        # existing settings files don't break.
        ebid = float(self.settings.get("edge_bid", 3.0))
        eask = float(self.settings.get("edge_ask", 3.0))
        if ebid < 1.0:
            ebid *= 100
        if eask < 1.0:
            eask *= 100
        # Spec: cent values, tenth-of-cent precision, half-cent stepper.
        self.edge_bid_input = self._make_compact_double_spin(
            "Edge Bid (¢)", ebid, 0.0, 50.0, 0.5, decimals=1)
        self.edge_ask_input = self._make_compact_double_spin(
            "Edge Ask (¢)", eask, 0.0, 50.0, 0.5, decimals=1)
        self.size_bid_input = self._make_compact_int_spin(
            "Size Bid", int(self.settings.get("size_bid", 10)), 1, 10000)
        self.size_ask_input = self._make_compact_int_spin(
            "Size Ask", int(self.settings.get("size_ask", 10)), 1, 10000)
        self.max_pos_input = self._make_compact_int_spin(
            "Max Pos", int(self.settings.get("max_position", 50)), 1, 10000)
        # Repricing tolerance — in CENTS for the UI, stored as dollars
        # in the strategy.  An existing resting order is only repriced
        # when fair has moved AGAINST it by ≥ tolerance.  Sub-tolerance
        # fair drift leaves the order in place, reducing churn and
        # giving stale-but-favorable prints a chance to fill.
        tol_cents = float(self.settings.get("tolerance", 1.0))
        if tol_cents < 0.1:
            # Migrate old dollar-unit values stored before this UI
            # change (0.01 dollar → 1 cent).
            tol_cents *= 100
        self.tolerance_input = self._make_compact_double_spin(
            "Tolerance (¢)", tol_cents, 0.0, 20.0, 0.1, decimals=1)
        # Dwell — min seconds a resting quote lives before it may be
        # cancel-replaced (reprice churn damper; pulls unaffected).
        self.dwell_input = self._make_compact_double_spin(
            "Dwell (s)", float(self.settings.get("dwell_s", 1.0)),
            0.0, 10.0, 0.1, decimals=1)

        for w in (self.edge_bid_input, self.edge_ask_input,
                  self.size_bid_input, self.size_ask_input,
                  self.max_pos_input, self.tolerance_input,
                  self.dwell_input):
            sp.addLayout(w)
        sp.addStretch()

        for spin in (self.edge_bid_input.spin, self.edge_ask_input.spin,
                     self.size_bid_input.spin, self.size_ask_input.spin,
                     self.max_pos_input.spin, self.tolerance_input.spin,
                     self.dwell_input.spin):
            spin.valueChanged.connect(self._on_param_changed)

        self.settings_panel.setVisible(False)
        outer.addWidget(self.settings_panel)

        # Status line — small, at the very bottom.
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#5a6270;font-size:11px;")
        outer.addWidget(self.status_label)

        outer.addStretch()

    def _toggle_settings_panel(self):
        on = self.settings_toggle.isChecked()
        self.settings_panel.setVisible(on)
        self.settings_toggle.setText("▼  Settings" if on else "▶  Settings")

    def _make_metric(self, title: str, initial: str, color: str) -> QWidget:
        """Title-over-value card.  All metric cards share size so the row
        reads as a uniform stat strip."""
        w = QWidget()
        w.setObjectName("metric")
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("metricTitle")
        v.addWidget(t)
        val = QLabel(initial)
        val.setFont(QFont("Menlo", 22, QFont.Weight.Bold))
        val.setStyleSheet(f"color:{color};background:transparent;border:none;")
        v.addWidget(val)
        w.value = val
        return w

    def _make_dual_metric(self, title1: str, initial1: str, color1: str,
                          title2: str, initial2: str, color2: str) -> QWidget:
        """Two stacked title/value pairs inside one card.  Used for
        TIME LEFT + T (years) so both live in the same box."""
        w = QWidget()
        w.setObjectName("metric")
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(2)

        t1 = QLabel(title1)
        t1.setObjectName("metricTitle")
        v.addWidget(t1)
        v1 = QLabel(initial1)
        v1.setFont(QFont("Menlo", 20, QFont.Weight.Bold))
        v1.setStyleSheet(f"color:{color1};background:transparent;border:none;")
        v.addWidget(v1)

        t2 = QLabel(title2)
        t2.setObjectName("metricTitle")
        t2.setStyleSheet(
            "color:#5a6270;font-size:9px;font-weight:bold;"
            "letter-spacing:1px;background:transparent;margin-top:4px;")
        v.addWidget(t2)
        v2 = QLabel(initial2)
        v2.setFont(QFont("Menlo", 13, QFont.Weight.Bold))
        v2.setStyleSheet(f"color:{color2};background:transparent;border:none;")
        v.addWidget(v2)

        # Two named slots — refresh code writes to .top / .bottom.
        w.top = v1
        w.bottom = v2
        return w

    def _make_compact_double_spin(self, title: str, initial: float,
                                  minv: float, maxv: float, step: float,
                                  decimals: int = 3):
        """Smaller settings spinbox — used in the tucked-away settings panel."""
        v = QVBoxLayout()
        v.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("settingTitle")
        v.addWidget(t)
        sb = QDoubleSpinBox()
        sb.setRange(minv, maxv)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        sb.setValue(float(initial))
        sb.setMaximumWidth(90)
        v.addWidget(sb)
        v.spin = sb
        return v

    def _make_compact_int_spin(self, title: str, initial: int,
                               minv: int, maxv: int):
        v = QVBoxLayout()
        v.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("settingTitle")
        v.addWidget(t)
        sb = QSpinBox()
        sb.setRange(minv, maxv)
        sb.setValue(int(initial))
        sb.setMaximumWidth(90)
        v.addWidget(sb)
        v.spin = sb
        return v

    def _apply_toggle_style(self, on: bool):
        """Style the MM toggle button (QToolButton with menu popup)."""
        if on:
            self.toggle_btn.setText("MM  ON")
            self.toggle_btn.setStyleSheet("""
                QToolButton{background:#16a34a;color:#fff;font-weight:bold;
                            font-size:14px;letter-spacing:2px;border:none;
                            border-top-left-radius:23px;
                            border-bottom-left-radius:23px;
                            padding:0 18px;}
                QToolButton:hover{background:#15803d;}
                QToolButton::menu-button{background:#15803d;
                            border:none;
                            border-top-right-radius:23px;
                            border-bottom-right-radius:23px;
                            width:26px;}
                QToolButton::menu-button:hover{background:#166534;}
            """)
        else:
            self.toggle_btn.setText("MM  OFF")
            self.toggle_btn.setStyleSheet("""
                QToolButton{background:#1e2736;color:#5a6270;font-weight:bold;
                            font-size:14px;letter-spacing:2px;border:none;
                            border-top-left-radius:23px;
                            border-bottom-left-radius:23px;
                            padding:0 18px;}
                QToolButton:hover{background:#2d3a4d;color:#c8cdd5;}
                QToolButton::menu-button{background:#2d3a4d;
                            border:none;
                            border-top-right-radius:23px;
                            border-bottom-right-radius:23px;
                            width:26px;}
                QToolButton::menu-button:hover{background:#3d4d63;}
            """)

    def _build_mm_menu(self):
        """Construct the dropdown attached to the MM toggle.  Holds
        the always-on flag and the auto-off threshold presets."""
        menu = QMenu(self)

        self.auto_mm_action = QAction(
            "Always on (auto new market)", self)
        self.auto_mm_action.setCheckable(True)
        self.auto_mm_action.setChecked(
            bool(self.settings.get("auto_mm_on", False)))
        self.auto_mm_action.toggled.connect(self._on_auto_mm_toggled)
        menu.addAction(self.auto_mm_action)

        menu.addSeparator()
        header_action = QAction("Auto-off threshold:", self)
        header_action.setEnabled(False)
        menu.addAction(header_action)

        self._threshold_group = QActionGroup(self)
        self._threshold_group.setExclusive(True)
        current = int(self.settings.get("auto_mm_off_secs", 90))
        for label, secs in [("30s", 30), ("1:00", 60), ("1:30", 90),
                            ("2:00", 120), ("3:00", 180)]:
            a = QAction(f"    {label}", self)
            a.setCheckable(True)
            a.setChecked(secs == current)
            a.triggered.connect(
                lambda _checked, s=secs: self._set_auto_mm_threshold(s))
            self._threshold_group.addAction(a)
            menu.addAction(a)

        self.toggle_btn.setMenu(menu)

    def _on_auto_mm_toggled(self, checked: bool):
        """Persist + immediately evaluate auto-state when the flag flips."""
        self.settings["auto_mm_on"] = bool(checked)
        _save_settings(self.settings)
        # Reset the per-ticker armed marker so toggling the flag back
        # on re-arms an auto-on for the current ticker.
        self._auto_mm_done_ticker = None
        self._check_auto_mm()

    def _set_auto_mm_threshold(self, secs: int):
        self.settings["auto_mm_off_secs"] = int(secs)
        _save_settings(self.settings)

    def _check_auto_mm(self):
        """When always-on mode is on, manage the toggle as a function of
        time-to-expiry:

          - On a fresh market roll (ticker just changed) and T > threshold:
            auto-engage MM.  Tracked once per ticker via
            `_auto_mm_done_ticker` so the user's manual OFF after auto-on
            isn't fought.
          - Anytime T < threshold while engaged: auto-disengage.  This
            stays active in always-on mode, so a manual ON in the
            forbidden zone will immediately flip back to OFF.
        """
        if not self.settings.get("auto_mm_on", False):
            return
        if not self.market or not self.ticker or self.strike <= 0:
            return

        secs = seconds_to_close(self.market)
        threshold = int(self.settings.get("auto_mm_off_secs", 90))
        is_on = self.strategy is not None

        # Auto-OFF
        if is_on and secs < threshold:
            self.toggle_btn.setChecked(False)
            self._on_toggle_clicked()
            self.status_label.setText(
                f"[AutoMM] OFF — {int(secs)}s < {threshold}s threshold")
            return

        # Auto-ON, once per ticker
        if (not is_on
                and secs > threshold
                and getattr(self, "_auto_mm_done_ticker", None)
                    != self.ticker):
            self._auto_mm_done_ticker = self.ticker
            self.toggle_btn.setChecked(True)
            self._on_toggle_clicked()
            if self.strategy is not None:
                self.status_label.setText(
                    f"[AutoMM] ON — {self.ticker} ({int(secs)}s left)")

    def _apply_stylesheet(self):
        # Two-tier visual hierarchy: main panel (#mainPanel) lifted with
        # its own border/background so it dominates; settings panel
        # (#settingsPanel) uses a flatter style so it reads as secondary.
        self.setStyleSheet("""
        QMainWindow{background:#0b0f19;}
        QWidget{background:#0b0f19;color:#c8cdd5;font-size:12px;}
        QLabel{color:#c8cdd5;background:transparent;}
        QLabel#metricTitle{color:#5a6270;font-size:10px;font-weight:bold;
                           letter-spacing:1px;background:transparent;}
        QLabel#settingTitle{color:#5a6270;font-size:9px;font-weight:bold;
                            letter-spacing:1px;background:transparent;}
        QFrame#mainPanel{background:#11172240;border:1px solid #1e2736;
                         border-radius:10px;}
        QFrame#settingsPanel{background:transparent;border:1px solid #1a212e;
                             border-radius:6px;}
        QWidget#metric{background:#161c28;border:1px solid #1e2736;
                       border-radius:8px;}
        QComboBox{background:#161c28;color:#c8cdd5;border:1px solid #1e2736;
                  border-radius:6px;padding:6px 10px;min-width:140px;
                  font-size:13px;}
        QComboBox::drop-down{border:none;}
        QComboBox QAbstractItemView{background:#161c28;color:#c8cdd5;
                                    selection-background-color:#1e2736;}
        QDoubleSpinBox, QSpinBox{background:#161c28;color:#c8cdd5;
            border:1px solid #1e2736;border-radius:5px;padding:3px 6px;
            font-size:11px;min-height:18px;}
        QDoubleSpinBox::up-button, QSpinBox::up-button,
        QDoubleSpinBox::down-button, QSpinBox::down-button{
            background:#1e2736;border:none;width:14px;}
        QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
        QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover{
            background:#2d3a4d;}
        QTableWidget{background:#0e1320;color:#c8cdd5;
                     gridline-color:#1e2736;border:1px solid #1e2736;
                     border-radius:6px;}
        QHeaderView::section{background:#161c28;color:#5a6270;
                             padding:6px;border:none;font-weight:bold;
                             font-size:10px;letter-spacing:1px;}
        QTableWidget::item{padding:4px 6px;}
        """)

    # ------------------------------------------------------------------
    # Series change → swap feeds + rediscover
    # ------------------------------------------------------------------

    def _current_series(self) -> dict:
        idx = self.series_combo.currentIndex()
        if 0 <= idx < len(SERIES_15M):
            return SERIES_15M[idx]
        return SERIES_15M[0]

    def _on_series_changed(self, index: int):
        # Stop existing strategy + feeds before swapping.
        self._teardown_current_market()
        if self.price_feed:
            self.price_feed.stop()
            self.price_feed = None

        s = self._current_series()
        # Fresh vol estimators per series so we don't blend e.g. BTC ticks
        # into an ETH vol estimate.
        self.vol_est = RealizedVolEstimator(
            lookback_minutes=float(self.settings.get("vol_lookback_min", 30.0)),
            sample_seconds=float(self.settings.get("vol_sample_sec", 10.0)),
        )
        self.har_est = HARRVEstimator(
            coef_path=Path(__file__).resolve().parent / "settings" / "har_coefficients.json"
        )
        # Kick off the 25h candle seed in the background.  The forecast
        # stays None until the worker completes and seed_from_candles
        # fills the buffer; vol_est covers the gap.
        self._har_seed_worker = HarSeedWorker(s["coinbase_product"])
        self._har_seed_worker.finished_signal.connect(self._on_har_seed)
        self._har_seed_worker.start()

        self.price_feed = CryptoPriceFeed(self._on_price, s["coinbase_product"])
        self.price_feed.start()
        self._rediscover()

    def _on_har_seed(self, candles: list):
        """Slot called when HarSeedWorker finishes the Coinbase backfill."""
        if not candles:
            print("[HARSeed] no candles — HAR will use prior + live data only")
            return
        self.har_est.seed_from_candles(candles)
        print(f"[HARSeed] seeded {len(candles)} candles "
              f"(buffer={self.har_est.sample_count()})")

    # ------------------------------------------------------------------
    # Active-market lifecycle
    # ------------------------------------------------------------------

    def _rediscover(self):
        """Find the current open 15-min market for the selected series.

        Single-flight: if a previous discovery worker is still in flight
        we skip rather than stacking workers (the periodic timer + the
        expiry hot-path both call this).
        """
        prev = getattr(self, "_discover_worker", None)
        if prev is not None and prev.isRunning():
            return
        s = self._current_series()
        worker = DiscoverWorker(self.api, s["ticker"])
        worker.finished_signal.connect(self._on_market_found)
        worker.start()
        self._discover_worker = worker  # keep ref so it isn't GC'd mid-run

    def _on_market_found(self, market: dict | None):
        if market is None:
            self.status_label.setText("No active market — waiting…")
            return
        # If the active market hasn't actually changed, refresh strike/close
        # in case the API updated mid-window but otherwise leave things alone.
        new_ticker = market.get("ticker", "")
        if self.market and self.market.get("ticker") == new_ticker:
            self.market = market
            # Backfill strike if a prior partial response left it at 0.
            # Without this, the only way to recover a missing strike was
            # to flip series and force a full teardown — observed in the
            # field on certain mid-roll API responses.
            if not self.strike or self.strike <= 0:
                self.strike = parse_strike(market)
            return
        # Different market — tear down old, wire up new.
        self._teardown_current_market()
        self.market = market
        self.ticker = new_ticker
        self.strike = parse_strike(market)
        self.ticker_label.setText(self.ticker)
        self.status_label.setText(f"Tracking {self.ticker}")

        # Seed bid/ask from the market dict so we don't sit at "--"
        # while waiting for the first WS update.
        try:
            self.yes_bid = float(market.get("yes_bid", 0) or 0) / 100.0
        except (TypeError, ValueError):
            self.yes_bid = 0.0
        try:
            self.yes_ask = float(market.get("yes_ask", 0) or 0) / 100.0
        except (TypeError, ValueError):
            self.yes_ask = 0.0

        # Reset position when rolling to a new market — each window is
        # its own settlement, no carry-over.  The seed worker below
        # will overwrite both fields immediately if Kalshi reports a
        # non-zero position (e.g. app restart while position is open).
        self._position = 0
        self._avg_entry = 0.0

        # Start WS feed for this single ticker.
        self.ws_feed = KalshiWsFeed(
            self.api, on_update=self._on_ws_update,
            on_fill=self._on_ws_fill,
            on_fill_raw=self._osm_on_fill_raw,
        )
        self.ws_feed.start([self.ticker])

        # Seed position + avg entry from REST so restarts (or any
        # external fills since last sync) are reflected.
        self._seed_position_from_rest()

    def _seed_position_from_rest(self):
        """Kick off a worker that fetches the current position + replays
        fills for self.ticker, then writes the result into local state.
        Single-flight: if a seed is already in progress, skip."""
        if not self.ticker:
            return
        prev = getattr(self, "_position_worker", None)
        if prev is not None and prev.isRunning():
            return
        worker = PositionSeedWorker(self.api, self.ticker)
        worker.finished_signal.connect(self._on_position_seeded)
        worker.start()
        self._position_worker = worker

    def _on_position_seeded(self, position: int, avg_entry: float):
        """Apply REST-derived position/avg-entry to local state.

        WS fills are the live source of truth — if any have arrived
        between worker start and finish we'd race them.  Simple
        defense: only apply if the WS hasn't already moved us off
        flat.  If you want strict authority from REST, drop the
        position==0 guard."""
        if self._position != 0:
            return
        self._position = position
        self._avg_entry = avg_entry
        # v2: OSM owns position.  If it's already running (always-on
        # auto-engage racing this REST fetch on restart), push the seed
        # through its queue — applied only if no WS fills arrived yet.
        if self.osm:
            self.osm.seed_position(position)
        if position != 0:
            print(f"[Seed] {self.ticker} pos={position:+d} "
                  f"avg={avg_entry*100:.1f}¢")

    def _teardown_current_market(self):
        # Stop strategy + WS for the previously-tracked market.
        if self.strategy:
            self.strategy.stop()
            self.strategy = None
            # Reflect the forced-off state in the toggle.
            self.toggle_btn.setChecked(False)
            self._apply_toggle_style(False)
        if self.osm:
            self.osm.stop()
            self.osm = None
        if self.ws_feed:
            self.ws_feed.stop()
            self.ws_feed = None
        self.market = None
        self.ticker = ""
        self.strike = 0.0
        self.yes_bid = 0.0
        self.yes_ask = 0.0

    # ------------------------------------------------------------------
    # Feed callbacks
    # ------------------------------------------------------------------

    def _on_price(self, price: float, bid: float = 0.0, ask: float = 0.0):
        """Coinbase WS tick — runs on the feed thread."""
        if price <= 0:
            return
        self.spot = price
        self.vol_est.on_price(price)
        self.har_est.on_price(price)
        self._recompute_and_trade()

    def _on_ws_update(self, ticker: str, yes_bid: float, yes_ask: float,
                      bid_size: int = 0, ask_size: int = 0):
        """Kalshi WS book update — runs on WS thread."""
        if ticker != self.ticker:
            return
        self.yes_bid = yes_bid
        self.yes_ask = yes_ask
        self.bid_size = bid_size
        self.ask_size = ask_size
        self._recompute_and_trade()

    def _osm_on_fill_raw(self, msg: dict):
        """Raw fill-channel forward to OSM.  No-op if OSM isn't running
        (defensive — OSM is rebuilt at every market roll)."""
        if self.osm is not None:
            self.osm.on_fill(msg)

    def _on_ws_fill(self, ticker: str, action: str, side: str,
                    price: float, count: int):
        """Kalshi WS fill — update position + avg entry + persist to DB."""
        if ticker != self.ticker:
            return

        # Yes-equivalent signed delta:
        #   yes side  + buy  → +qty (acquiring YES)
        #   yes side  + sell → -qty (releasing YES)
        #   no  side  + buy  → -qty (long NO = short YES)
        #   no  side  + sell → +qty
        if (side == "yes" and action == "buy") or \
           (side == "no" and action == "sell"):
            delta = float(count)
        else:
            delta = -float(count)

        prev_pos = self._position
        new_pos = prev_pos + int(delta)
        # Average entry math — same as PositionManager.compute_trade_view.
        # Quote prices for NO buys / NO sells are flipped to the YES side
        # (1 - p) so avg_entry always lives in YES dollar space.
        yes_price = price if side == "yes" else (1.0 - price)
        if prev_pos == 0:
            self._avg_entry = yes_price
        elif (prev_pos > 0 and delta > 0) or (prev_pos < 0 and delta < 0):
            # Adding to same direction — weighted average
            self._avg_entry = (
                (self._avg_entry * abs(prev_pos) + yes_price * abs(delta))
                / abs(new_pos)
            )
        elif new_pos == 0:
            self._avg_entry = 0.0
        elif abs(delta) > abs(prev_pos):
            # Flipping past zero — new position opens at fill price
            self._avg_entry = yes_price
        # else: reducing same direction — avg unchanged
        self._position = new_pos

        if self.strategy:
            self.strategy.position = self._position
            self.strategy.on_fill(action=action, price=price,
                                  count=count, side=side)

        # Note: persistence is owned by the standalone `recorder.py`
        # process, which listens to user_orders WS and writes the same
        # per-day DB files PositionManager reads.  The app no longer
        # records fills directly — start `python3 recorder.py` in a
        # second terminal alongside this app for forensic capture.

        print(f"[Fill] {ticker} {action} {side} x{count} @ {price*100:.1f}¢ "
              f"pos={self._position}")

    def _recompute_and_trade(self):
        """Compute theo from current spot + vol + time-to-expiry, then
        feed the strategy.  Runs on the WS / price-feed threads."""
        if not self.market:
            return
        # Track recompute cadence — EMA of ms since the last call.
        # Both Coinbase ticker WS and Kalshi book updates drive this,
        # so the number reflects total theo-recompute rate.
        now = time.monotonic()
        if self._theo_last_ts is not None:
            dt_ms = (now - self._theo_last_ts) * 1000.0
            if self._theo_dt_ms == 0.0:
                self._theo_dt_ms = dt_ms
            else:
                self._theo_dt_ms = 0.8 * self._theo_dt_ms + 0.2 * dt_ms
        self._theo_last_ts = now

        # Prefer HAR-RV forecast; fall back to the simple rolling RV
        # while HAR is still seeding (first ~25h candle backfill or if
        # the seed worker errored out).
        sigma = self.har_est.get_annualized_vol()
        if sigma is None:
            sigma = self.vol_est.get_annualized_vol()
        secs = seconds_to_close(self.market)
        theo = compute_theo(self.spot, self.strike, sigma or 0.0, secs)
        self._last_theo = theo
        self._last_sigma = sigma
        if self.strategy and theo is not None:
            self.strategy.update_bbo((self.yes_bid, self.yes_ask))
            self.strategy.update_theo(theo)

    # ------------------------------------------------------------------
    # Strategy start/stop
    # ------------------------------------------------------------------

    def _on_toggle_clicked(self):
        """Single on/off switch.  `isChecked()` reflects the new state."""
        if self.toggle_btn.isChecked():
            if not self.market or not self.ticker or self.strike <= 0:
                self.toggle_btn.setChecked(False)
                self.status_label.setText("No market to quote yet")
                return
            if self.strategy:
                self._apply_toggle_style(True)
                return  # already running, just normalize visual state
            # Edges in the UI are in cents; strategy works in dollars.
            # Construct OSM first (with seeded position + max_position
            # so the cap is correct from the first message), backfill
            # strategy queue reference after Strategy2 exists.
            self.osm = OSM(
                ticker=self.ticker,
                tolerance=self.tolerance_input.spin.value() / 100.0,
                api=self.api,
                max_position=self.max_pos_input.spin.value(),
                position=self._position,
                strategy_queue=None,
                gateway=self.gateway,
            )
            self.strategy = Strategy2(
                ticker=self.ticker, strike=self.strike,
                edge_bid=self.edge_bid_input.spin.value() / 100.0,
                edge_ask=self.edge_ask_input.spin.value() / 100.0,
                size_bid=self.size_bid_input.spin.value(),
                size_ask=self.size_ask_input.spin.value(),
                osm=self.osm,
            )
            self.osm.strategy_queue = self.strategy.queue
            self.osm.start()
            self.strategy.start()
            self._apply_toggle_style(True)
            self.status_label.setText(f"MM active on {self.ticker}")
            self._save_current_params()
        else:
            if self.strategy:
                self.strategy.stop()
                self.strategy = None
            if self.osm:
                self.osm.stop()
                self.osm = None
            self._apply_toggle_style(False)
            self.status_label.setText("MM stopped")

    def _on_param_changed(self, _value):
        """Push parameter changes to a running strategy without restart."""
        self._save_current_params()
        if self.strategy:
            self.strategy.update_params(
                edge_bid=self.edge_bid_input.spin.value() / 100.0,
                edge_ask=self.edge_ask_input.spin.value() / 100.0,
                size_bid=self.size_bid_input.spin.value(),
                size_ask=self.size_ask_input.spin.value(),
                max_position=self.max_pos_input.spin.value(),
                tolerance=self.tolerance_input.spin.value() / 100.0,
                dwell_s=self.dwell_input.spin.value(),
            )

    def _save_current_params(self):
        self.settings["edge_bid"] = self.edge_bid_input.spin.value()
        self.settings["edge_ask"] = self.edge_ask_input.spin.value()
        self.settings["size_bid"] = self.size_bid_input.spin.value()
        self.settings["size_ask"] = self.size_ask_input.spin.value()
        self.settings["max_position"] = self.max_pos_input.spin.value()
        self.settings["tolerance"] = self.tolerance_input.spin.value()
        self.settings["dwell_s"] = self.dwell_input.spin.value()
        _save_settings(self.settings)

    # ------------------------------------------------------------------
    # UI refresh
    # ------------------------------------------------------------------

    def _refresh_ui(self):
        # Token budget — green when healthy, yellow under half, red when
        # a create (100) is no longer affordable.
        toks = self.gateway.tokens_remaining()
        color = "#22c55e" if toks >= 300 else ("#facc15" if toks >= 100 else "#dc2626")
        self.tokens_label.setText(f"{toks:.0f}/600")
        self.tokens_label.setStyleSheet(
            f"color:{color};background:transparent;border:none;")

        # Spot / Strike
        self.spot_card.value.setText(
            f"${self.spot:,.2f}" if self.spot > 0 else "--")
        self.strike_card.value.setText(
            f"${self.strike:,.2f}" if self.strike > 0 else "--")

        # Combined TIME LEFT + T (years) card.
        # T uses the same SECONDS_PER_YEAR denominator as theo_engine, so
        # the value shown is exactly what feeds the N(d2) formula.
        if self.market:
            raw_secs = seconds_to_close(self.market)
            secs = max(raw_secs, 0.0)
            mm = int(secs // 60)
            ss = int(secs % 60)
            self.time_card.top.setText(f"{mm:02d}:{ss:02d}")
            t_years = secs / (365.25 * 24 * 3600)
            # Plain decimal with enough digits to be useful at the 15-min
            # scale.  For 15 min (~900s) T ≈ 0.0000285 — 7 decimals lands
            # the leading digit visibly.
            self.time_card.bottom.setText(f"{t_years:.7f}")
            color = "#ef4444" if secs <= 60 else "#22c55e"
            self.time_card.top.setStyleSheet(
                f"color:{color};background:transparent;border:none;")
            # Hot-path rediscover: the moment the countdown hits zero,
            # kick a discovery so we roll to the next 15-min window
            # without waiting up to 15s for the next timer tick.  The
            # single-flight guard in _rediscover keeps this from stacking
            # workers across consecutive UI ticks while the API call is
            # in flight.
            if raw_secs <= 0:
                self._rediscover()
        else:
            self.time_card.top.setText("--:--")
            self.time_card.bottom.setText("--")

        # Yes bid / ask in CENTS — 1 decimal so tenths-of-cent ticks
        # are visible, with the resting size at that level in parens.
        # Internal `self.yes_bid` stays in dollar units.
        bid_sz = f" ({self.bid_size})" if self.bid_size > 0 else ""
        ask_sz = f" ({self.ask_size})" if self.ask_size > 0 else ""
        self.bid_card.value.setText(
            f"{self.yes_bid * 100:.1f}¢{bid_sz}" if self.yes_bid > 0 else "--")
        self.ask_card.value.setText(
            f"{self.yes_ask * 100:.1f}¢{ask_sz}" if self.yes_ask > 0 else "--")

        # Theo in cents
        if self._last_theo is not None:
            self.theo_card.value.setText(f"{self._last_theo * 100:.1f}¢")
        else:
            self.theo_card.value.setText("--")
        # Recompute cadence — EMA inter-arrival of _recompute_and_trade.
        # Stale if the last recompute was >2s ago.
        if (self._theo_last_ts is not None
                and (time.monotonic() - self._theo_last_ts) < 2.0
                and self._theo_dt_ms > 0):
            self.theo_rate_label.setText(f"{self._theo_dt_ms:.0f} ms")
        else:
            self.theo_rate_label.setText("-- ms")

        # Realized vol — as percent
        if self._last_sigma is not None:
            self.vol_card.value.setText(f"{self._last_sigma * 100:.1f}%")
        else:
            samples = self.vol_est.sample_count()
            self.vol_card.value.setText(f"warming ({samples})")
        self._refresh_vol_tooltip()

        # Auto-MM lifecycle (always-on mode, threshold-based shutoff)
        self._check_auto_mm()

        # Position card — pos count plus avg-entry-of-open-position in
        # parentheses, e.g. "+12 (53.4¢)".  Avg entry is the weighted
        # YES-equivalent fill price of the currently-open exposure; goes
        # away when flat.
        if self._position > 0:
            color = "#22c55e"
        elif self._position < 0:
            color = "#ef4444"
        else:
            color = "#c8cdd5"
        if self._position == 0:
            text = "0"
        elif self._avg_entry > 0:
            text = f"{self._position:+d} ({self._avg_entry * 100:.1f}¢)"
        else:
            text = f"{self._position:+d}"
        self.position_card.value.setText(text)
        # Slightly smaller font when the parenthetical is present so it
        # fits comfortably in the card width.
        if self._position != 0 and self._avg_entry > 0:
            self.position_card.value.setFont(
                QFont("Menlo", 16, QFont.Weight.Bold))
        else:
            self.position_card.value.setFont(
                QFont("Menlo", 22, QFont.Weight.Bold))
        self.position_card.value.setStyleSheet(
            f"color:{color};background:transparent;border:none;")

        # Cash + portfolio value — both come straight from Kalshi's
        # /portfolio/balance.  Portfolio = balance + portfolio_value
        # (Kalshi's server-side MtM of open positions across all
        # markets, not just the active ticker).
        if self._balance_cents < 0:
            self.cash_label.setText("--")
            self.portfolio_label.setText("--")
        else:
            cash = self._balance_cents / 100.0
            portfolio = (self._balance_cents + self._portfolio_value_cents) / 100.0
            self.cash_label.setText(f"${cash:,.2f}")
            self.portfolio_label.setText(f"${portfolio:,.2f}")

        # Live orders — read from strategy state on the same 250ms tick.
        self._render_orders()

    # ------------------------------------------------------------------
    # Portfolio + live orders refresh
    # ------------------------------------------------------------------

    def _refresh_balance(self):
        if self._balance_worker is not None and self._balance_worker.isRunning():
            return
        self._balance_worker = BalanceWorker(self.api)
        self._balance_worker.finished_signal.connect(self._on_balance_received)
        self._balance_worker.start()

    def _on_balance_received(self, balance_cents: int,
                             portfolio_value_cents: int):
        if balance_cents >= 0:
            self._balance_cents = balance_cents
            self._portfolio_value_cents = portfolio_value_cents

    def _refresh_vol_tooltip(self):
        """Rebuild the HAR-RV hover tooltip with the current per-horizon
        RV values and the active coefficients.  Called every UI tick."""
        b = self.har_est.horizon_breakdown()
        def pct(x): return f"{x * 100:.1f}%" if x is not None else "--"
        def coef(k): return f"{b['coef'][k]:+.4f}"

        if b["samples"] < 1441:
            status = f"warming: {b['samples']} / 1441 closes"
        else:
            status = f"forecast next-15m RV = {pct(b['forecast'])}"

        meta_parts = [f"coefficients: {b['coef_source']}"]
        if b["fit_at"]:
            meta_parts.append(f"fit {b['fit_at'][:10]}")
        if b["r2_test"] is not None:
            meta_parts.append(f"R² OOS {b['r2_test']:.2f}")
        if b["n_train"] is not None:
            meta_parts.append(f"n={b['n_train']}")
        meta = " · ".join(meta_parts)

        # Plain text with \n line breaks — most reliable across Qt
        # tooltip rendering paths.  Qt auto-detects plain vs rich text.
        estimator = b.get("estimator", "cc_1min")
        if estimator == "parkinson":
            est_line = "estimator: Parkinson (high-low), 1-min buckets"
        else:
            est_line = f"estimator: {estimator}, 1-min sampling"
        text = (
            "HAR-RV\n"
            f"{est_line}\n"
            f"{status}\n"
            "\n"
            f"  RV 15m   {pct(b['rv_15m'])}\n"
            f"  RV 30m   {pct(b['rv_30m'])}\n"
            f"  RV 4h    {pct(b['rv_4h'])}\n"
            f"  RV 24h   {pct(b['rv_24h'])}\n"
            "\n"
            f"  β0     {coef('beta0')}\n"
            f"  β 15m  {coef('beta_15')}\n"
            f"  β 30m  {coef('beta_30')}\n"
            f"  β 4h   {coef('beta_4h')}\n"
            f"  β 24h  {coef('beta_24h')}\n"
            "\n"
            f"{meta}"
        )
        self.vol_info_label.setToolTip(text)
        self.vol_card.setToolTip(text)

    def _render_orders(self):
        """Render the 2-row orders panel directly from strategy state.

        Read on every UI tick (250ms).  When MM is off or no order is
        resting on a side, the corresponding row shows `--`.  Source of
        truth is `self.strategy.current_{buy,sell}_price` /
        `resting_{buy,sell}_count` — these flip the instant the place
        callback runs, so the panel reflects placements with no REST
        round-trip.
        """
        strat = self.strategy

        def _fmt_age(age_s: float | None) -> str:
            if age_s is None or age_s < 0:
                return "--"
            if age_s < 10:
                return f"{age_s:.1f}s"
            if age_s < 120:
                return f"{age_s:.0f}s"
            return f"{age_s / 60:.0f}m{age_s % 60:02.0f}s"

        def _set(row: int, side_label: str, side_color: str,
                 price_dollars: float | None, size: int,
                 age_s: float | None = None):
            price_text = (f"{price_dollars * 100:.1f}¢"
                          if price_dollars and price_dollars > 0 else "--")
            size_text = f"{size}" if size > 0 else "--"
            cells = [
                (side_label, side_color, True),
                (price_text, "#facc15", False),
                (size_text, "#c8cdd5", False),
                (_fmt_age(age_s), "#8b95a5", False),
            ]
            for c, (text, color, bold) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(QColor(color))
                if bold:
                    f = item.font(); f.setBold(True); item.setFont(f)
                self.orders_table.setItem(row, c, item)

        if strat is None:
            _set(0, "BUY",  "#22c55e", None, 0)
            _set(1, "SELL", "#ef4444", None, 0)
            return

        # Strategy2 keeps resting state in OSM; legacy Strategy keeps
        # it on itself.  Branch on the presence of `osm` so both
        # implementations render without needing isinstance checks.
        osm = getattr(strat, "osm", None)
        if osm is not None:
            now = time.time()
            def _age(q):
                return (now - q.placed_at) if (q and q.placed_at > 0) else None
            bid_q = osm.resting_bid
            ask_q = osm.resting_ask
            buy_price = bid_q.price if bid_q else None
            buy_size = bid_q.size if bid_q else 0
            sell_price = ask_q.price if ask_q else None
            sell_size = ask_q.size if ask_q else 0
            buy_age, sell_age = _age(bid_q), _age(ask_q)
        else:
            buy_price = strat.current_buy_price
            buy_size = strat.resting_buy_count
            sell_price = strat.current_sell_price
            sell_size = strat.resting_sell_count
            buy_age = sell_age = None

        _set(0, "BUY",  "#22c55e", buy_price,  buy_size,  buy_age)
        _set(1, "SELL", "#ef4444", sell_price, sell_size, sell_age)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.strategy:
            self.strategy.stop()
            self.strategy = None
        if self.osm:
            self.osm.stop()
            self.osm = None
        if self.ws_feed:
            self.ws_feed.stop()
        if self.price_feed:
            self.price_feed.stop()
        try:
            self.api.shutdown()
        except Exception:
            pass
        event.accept()


def main():
    import argparse
    from pathlib import Path
    from tools.tee_log import install
    install("aston", Path(__file__).resolve().parents[1] / "logs" / "Aston")

    ap = argparse.ArgumentParser()
    args, qt_argv = ap.parse_known_args()
    print("[Aston] launching with Strategy v2")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(qt_argv)
    app.setApplicationName("Aston")
    # macOS Dock hover label is read from the bundle's CFBundleName, not
    # QApplication.setApplicationName — running a loose .py would otherwise
    # show "Python".  Best-effort override; silently no-op if PyObjC missing.
    try:
        from Foundation import NSBundle  # type: ignore
        bundle = NSBundle.mainBundle()
        if bundle:
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info is not None:
                info["CFBundleName"] = "Aston"
                info["CFBundleDisplayName"] = "Aston"
    except Exception:
        pass

    icon_path = Path(__file__).resolve().parent / "tools/icon.png"
    # Self-heal: if the user hasn't dropped an icon.png next to app.py,
    # generate a default Aston-style emblem and save it.  Subsequent
    # launches pick up the saved file, and the user can overwrite it
    # with their own image at any time.
    if not icon_path.exists():
        try:
            _generate_default_icon(icon_path)
            print(f"[Aston] Generated default icon at {icon_path}")
        except Exception as e:
            print(f"[Aston] Could not generate default icon: {e}")

    if icon_path.exists():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)
        # macOS Dock icon comes from NSApplication.applicationIconImage,
        # not Qt's window icon.  Push the same PNG through PyObjC so the
        # Dock matches the title-bar / taskbar icon.
        try:
            from AppKit import NSApplication, NSImage  # type: ignore
            ns_img = NSImage.alloc().initByReferencingFile_(str(icon_path))
            NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
        except Exception:
            pass

    window = AstonApp()
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show()
    sys.exit(app.exec())


def _generate_default_icon(path: Path):
    """Draw a fallback Aston-style emblem and save to PNG.

    Square 512×512 with alpha.  Dark rounded-square base, chrome rim,
    silver serif 'A' centered, two horizontal wing bars in Aston green.
    Looks reasonable in Dock + title-bar at any size.  Designed to be
    overwritten by a user-supplied icon.png in the same directory.
    """
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import (QColor, QFont, QLinearGradient, QPainter, QPen,
                              QPixmap)

    size = 512
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    # Rounded-square background — vertical gradient
    grad = QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0.0, QColor("#0e131c"))
    grad.setColorAt(1.0, QColor("#1e2736"))
    p.setBrush(grad)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(12, 12, size - 24, size - 24, 72, 72)

    # Chrome rim
    p.setBrush(Qt.GlobalColor.transparent)
    p.setPen(QPen(QColor("#94a3b8"), 5))
    p.drawRoundedRect(12, 12, size - 24, size - 24, 72, 72)

    # Stylized 'A' — serif, bold, silver
    font = QFont("Times New Roman", int(size * 0.55))
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor("#e2e8f0"))
    p.drawText(QRect(0, int(size * 0.05), size, int(size * 0.74)),
               int(Qt.AlignmentFlag.AlignCenter), "A")

    # Two horizontal wing bars below — Aston green
    p.setPen(QPen(QColor("#16a34a"), 10))
    y1 = int(size * 0.84)
    p.drawLine(int(size * 0.18), y1, int(size * 0.82), y1)
    y2 = int(size * 0.91)
    p.drawLine(int(size * 0.28), y2, int(size * 0.72), y2)

    p.end()
    pixmap.save(str(path), "PNG")


if __name__ == "__main__":
    main()
