"""
4RunnerApp 2.0 — Above/Below Theo Viewer

Displays Deribit-implied probabilities for "above K" at every Kalshi
above/below strike (KXBTCD series). Theos come from the first derivative
of Deribit call prices:

    P(S > K) = -dC/dK

Shows live Kalshi Yes Bid / Yes Ask alongside the Deribit theo,
plus the two Deribit option strikes used for each computation.

Features:
    - Crypto dropdown (BTC, ETH, etc.)
    - Event dropdown (auto-discovers weekly events)
    - Live spot price from Coinbase
    - Live Kalshi bid/ask via websocket
    - Table: Strike | Yes Bid | Yes Ask | Theo | Deribit K_low | K_high | dC/dK
    - Auto-refresh Deribit data every 60s
"""

import sys
import json
import math
import time
import signal
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

_norm = NormalDist()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QLineEdit, QDialog, QFormLayout, QMenu, QMessageBox,
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from kalshi_api import KalshiAPI
from market_discovery import discover_events_for_series, parse_strike, display_strike
from btc_price_feed import CryptoPriceFeed
from ws_feed import KalshiWsFeed
from deribit_vol import (
    DeribitBracketPricer, DeribitWsFeed, find_deribit_expiry,
    list_deribit_expiries, KALSHI_TO_DERIBIT_CURRENCY,
)
from strategy import Strategy


# =============================================================================
# Config
# =============================================================================

_SERIES_FILE = Path(__file__).parent / "series.json"
_STRATEGY_FILE = Path(__file__).parent / "strategy_params.json"
_AUTO_EDGE_FILE = Path(__file__).parent / "auto_edge_params.json"

def _implied_vol_quadratic(price: float, spot: float, strike: float,
                           T: float, r: float = 0.0) -> float:
    """Closed-form quadratic IV for a binary above option.

    Given P(above K) = N(d2), invert for σ:
        x = N⁻¹(P),  m = ln(S/K) + rT
        u² + 2xu − 2m = 0  →  u = −x + √(x² + 2m)
        σ = u / √T

    Returns IV as a decimal (e.g. 0.65 for 65%), or 0 if unsolvable.
    """
    if price <= 0.01 or price >= 0.99 or spot <= 0 or strike <= 0 or T <= 0:
        return 0.0
    try:
        x = _norm.inv_cdf(price)
        m = math.log(spot / strike) + r * T
        disc = x * x + 2 * m
        if disc < 0:
            return 0.0
        sqrt_disc = math.sqrt(disc)
        # Two roots: u = -x ± √(disc). Pick the smallest positive root.
        u1 = -x + sqrt_disc
        u2 = -x - sqrt_disc
        candidates = [u for u in (u1, u2) if u > 0]
        if not candidates:
            return 0.0
        u = min(candidates)
        return u / math.sqrt(T)
    except Exception:
        return 0.0


def _load_series() -> list[dict]:
    try:
        with open(_SERIES_FILE) as f:
            return json.load(f)
    except Exception:
        return [{"ticker": "KXBTCD", "name": "Bitcoin", "coinbase_product": "BTC-USD"}]

def _load_strategy_params() -> dict:
    """Load saved strategy params. Returns {strike: {edge, size, max_position, ...}}."""
    try:
        with open(_STRATEGY_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_strategy_params(params: dict):
    """Persist strategy params to disk."""
    with open(_STRATEGY_FILE, "w") as f:
        json.dump(params, f, indent=2)

def _load_auto_edge_params() -> dict:
    try:
        with open(_AUTO_EDGE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_auto_edge_params(params: dict):
    with open(_AUTO_EDGE_FILE, "w") as f:
        json.dump(params, f, indent=2)

CRYPTO_SERIES = _load_series()


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


class DeribitDiscoverWorker(QThread):
    """Discovers Deribit instruments via REST (single fast call)."""
    finished = pyqtSignal(str, list)   # (expiry_str, instrument_names)
    error = pyqtSignal(str)

    def __init__(self, pricer, expiry_str):
        super().__init__()
        self.pricer = pricer
        self.expiry_str = expiry_str

    def run(self):
        try:
            names = self.pricer.discover_instruments(self.expiry_str)
            if len(names) >= 5:
                self.finished.emit(self.expiry_str, names)
            else:
                self.error.emit(f"Only {len(names)} calls for {self.expiry_str}")
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# IV Smile Window
# =============================================================================

def implied_vol_from_prob(prob: float, S: float, K: float, T: float,
                          r: float = 0.043) -> float:
    """Back out implied volatility from a binary option probability.

    Given P(S>K) = N(d2), solve for sigma using bisection.
    Returns IV as decimal (e.g. 0.65 for 65%), or 0 if unsolvable.
    """
    import math
    if T <= 0 or S <= 0 or K <= 0 or prob <= 0.01 or prob >= 0.99:
        return 0.0
    log_sk = math.log(S / K)

    def bs_prob(sigma):
        sqrt_T = math.sqrt(T)
        d2 = (log_sk + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        return 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = bs_prob(mid)
        if p > prob:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


class IVSmileWindow(QWidget):
    """Separate window showing the fitted vol smile curve + data points used."""

    def __init__(self, pricer, app_ref):
        super().__init__()
        self.pricer = pricer
        self.app_ref = app_ref
        self.setWindowTitle("IV Smile")
        self.resize(700, 450)
        self.setStyleSheet("background:#0b0f19;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.fig = Figure(figsize=(7, 4), facecolor="#0b0f19")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Refresh every 10s
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(10_000)

        self.update_chart()

    def update_chart(self):
        ax = self.ax
        ax.clear()

        app = self.app_ref
        spot = app.spot_price
        if spot <= 0 or not app.display_strikes:
            self.canvas.draw()
            return

        # Compute T
        T = 0.0
        if app.current_event:
            close_time = app.current_event.get("close_time", "")
            if close_time:
                try:
                    close_utc = datetime.fromisoformat(
                        close_time.replace("Z", "+00:00"))
                    T = max((close_utc - datetime.now(tz=timezone.utc)
                             ).total_seconds() / (365.25 * 24 * 3600), 0.0)
                except Exception:
                    pass
        if T <= 0:
            self.canvas.draw()
            return

        # Compute mid IV for all strikes, track which are nearby (within 4%)
        nearby_strikes = []
        nearby_ivs = []
        for raw, disp in zip(app.all_strikes, app.all_display_strikes):
            data = app.market_data.get(raw, {})
            bid = data.get("yes_bid", 0)
            ask = data.get("yes_ask", 0)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            iv = _implied_vol_quadratic(mid, spot, disp, T,
                                        app.pricer.risk_free_rate)
            if iv <= 0:
                continue
            otm = abs(disp / spot - 1)
            if otm < app._smile_otm_pct:
                nearby_strikes.append(disp)
                nearby_ivs.append(iv * 100.0)

        # Reject IV outliers using IQR (same logic as _refit_vol_smile)
        if nearby_ivs:
            ivs_arr = np.array(nearby_ivs)
            q1, q3 = np.percentile(ivs_arr, [25, 75])
            iqr = q3 - q1
            iv_low = q1 - 1.5 * iqr
            iv_high = q3 + 1.5 * iqr
            filtered = [(k, iv) for k, iv in zip(nearby_strikes, nearby_ivs)
                        if iv_low <= iv <= iv_high]
            nearby_strikes = [p[0] for p in filtered]
            nearby_ivs = [p[1] for p in filtered]

        # Plot the data points used for fitting
        if nearby_strikes:
            ax.scatter(nearby_strikes, nearby_ivs, color="#facc15",
                       s=30, zorder=3, label="Mid IV (fit points)")

        # Plot the fitted curve
        a, b, c = app._smile_coeffs
        if a != 0 or b != 0 or c != 0:
            k_min = min(nearby_strikes) if nearby_strikes else spot * 0.96
            k_max = max(nearby_strikes) if nearby_strikes else spot * 1.04
            k_range = np.linspace(k_min, k_max, 200)
            fitted_iv = (a * k_range**2 + b * k_range + c) * 100.0
            ax.plot(k_range, fitted_iv, color="#8b5cf6", linewidth=2,
                    label="Fitted smile")

        # Mark spot
        if spot > 0:
            ax.axvline(spot, color="#ef4444", linestyle="--",
                       linewidth=1, alpha=0.4, label="Spot")

        ax.legend(facecolor="#141923", edgecolor="#1e2736",
                  labelcolor="#c8cdd5", fontsize=9, loc="upper right")

        ax.set_xlabel("Strike", color="#5a6270", fontsize=10)
        ax.set_ylabel("IV (%)", color="#5a6270", fontsize=10)
        ax.set_title("Fitted Vol Smile", color="#c8cdd5", fontsize=12)
        ax.set_facecolor("#0b0f19")
        ax.tick_params(colors="#5a6270", labelsize=9)
        ax.spines["bottom"].set_color("#1e2736")
        ax.spines["left"].set_color("#1e2736")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#1e2736", alpha=0.5)

        self.fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()


# =============================================================================
# Strike Params Dialog
# =============================================================================

class StrikeParamsDialog(QDialog):
    """Popup to adjust bid/ask edge, bid/ask size, max position, tolerance."""

    def __init__(self, strike_label: str,
                 edge_bid: float, edge_ask: float,
                 size_bid: int, size_ask: int,
                 max_pos: int, tolerance: float = 0.01,
                 flatten_walk_interval: float = 0.0,
                 flatten_walk_step: float = 0.01,
                 parent=None, auto_edge: bool = False):
        super().__init__(parent)
        self.setWindowTitle(f"Params — {strike_label}")
        self.setFixedSize(260, 360)
        self.setStyleSheet(
            "QDialog{background:#0b0f19;}"
            "QLabel{color:#c8cdd5;font-size:12px;}"
            "QLineEdit{background:#141923;color:#c8cdd5;"
            "border:1px solid #1e2736;border-radius:3px;padding:4px 8px;}"
            "QPushButton{background:#1e2736;color:#c8cdd5;border:1px solid #2d3a4d;"
            "border-radius:3px;padding:6px 12px;}"
            "QPushButton:hover{background:#2d3a4d;}"
        )

        layout = QFormLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.edge_bid_input = QLineEdit(f"{edge_bid}")
        self.edge_ask_input = QLineEdit(f"{edge_ask}")
        if auto_edge:
            for inp in (self.edge_bid_input, self.edge_ask_input):
                inp.setEnabled(False)
                inp.setStyleSheet(
                    "QLineEdit{background:#0d1117;color:#5a6270;"
                    "border:1px solid #1e2736;border-radius:3px;padding:4px 8px;}"
                )
        self.size_bid_input = QLineEdit(f"{size_bid}")
        self.size_ask_input = QLineEdit(f"{size_ask}")
        self.max_pos_input = QLineEdit(f"{max_pos}")
        self.tolerance_input = QLineEdit(f"{tolerance}")
        self.flatten_interval_input = QLineEdit(f"{flatten_walk_interval}")
        self.flatten_step_input = QLineEdit(f"{flatten_walk_step}")

        edge_label = "Bid Edge (auto):" if auto_edge else "Bid Edge:"
        layout.addRow(edge_label, self.edge_bid_input)
        edge_label = "Ask Edge (auto):" if auto_edge else "Ask Edge:"
        layout.addRow(edge_label, self.edge_ask_input)
        layout.addRow("Bid Size:", self.size_bid_input)
        layout.addRow("Ask Size:", self.size_ask_input)
        layout.addRow("Max Pos:", self.max_pos_input)
        layout.addRow("Tolerance:", self.tolerance_input)
        layout.addRow("Walk Interval (s):", self.flatten_interval_input)
        layout.addRow("Walk Step ($):", self.flatten_step_input)

        btn = QPushButton("Apply")
        btn.clicked.connect(self.accept)
        layout.addRow(btn)

    def get_params(self):
        """Return params tuple or None."""
        try:
            return (float(self.edge_bid_input.text()),
                    float(self.edge_ask_input.text()),
                    int(self.size_bid_input.text()),
                    int(self.size_ask_input.text()),
                    int(self.max_pos_input.text()),
                    float(self.tolerance_input.text()),
                    float(self.flatten_interval_input.text()),
                    float(self.flatten_step_input.text()))
        except ValueError:
            return None


# =============================================================================
# Main Window
# =============================================================================

class AboveBelowApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("4Runner 2.0 — Above/Below Theos")
        self.resize(1000, 700)

        self.api = KalshiAPI()
        self.api.on_rate_limit = self._on_rate_limit
        self._rate_limit_remaining = None
        self._rate_limit_total = None
        self._rate_limit_hit_time = 0
        self.pricer = DeribitBracketPricer()
        self.pricer.risk_free_rate = 0.043  # ~4.3% annualised (T-bill rate)
        self.weekly_pricer = DeribitBracketPricer()
        self.weekly_pricer.risk_free_rate = 0.043
        self._deribit_discover_worker = None
        self._weekly_discover_worker = None
        self._discover_worker = None
        self.deribit_ws = None      # DeribitWsFeed
        self.weekly_deribit_ws = None  # Weekly DeribitWsFeed
        self._weekly_expiry_str = None
        self.iv_window = None       # IVSmileWindow

        self.events = []
        self.current_event = None
        self.strikes = []           # sorted raw strikes from tickers
        self.display_strikes = []   # rounded display values
        self.spot_price = 0.0
        self.spot_bid = 0.0
        self.spot_ask = 0.0
        self.price_feed = None

        # WS feed for live Kalshi bid/ask
        self.ws_feed = None
        # Map: raw_strike -> {"ticker": str, "yes_bid": float, "yes_ask": float}
        self.market_data = {}

        self._balance_cents = 0
        self.strategies = {}  # raw_strike -> Strategy instance
        self._stashed_strategies = {}   # event_ticker -> {raw_strike: Strategy}
        self._stashed_market_data = {}  # event_ticker -> {raw_strike: market_data}
        self._stashed_events = {}       # event_ticker -> event dict (for close_time)
        self._fill_flash = {}           # raw_strike -> monotonic timestamp of last fill
        self._portfolio_history = []    # [(datetime_utc, portfolio_dollars)]
        self._pnl_baseline: dict[float, float] = {}  # raw_strike -> PnL at app start

        # OTM% filter — only show strikes within this range
        self.otm_filter_pct: float = 1.5       # show strikes within ±1.5% OTM
        self.otm_hysteresis: float = 0.3        # keep showing until OTM exceeds threshold + 0.3%
        self.visible_strikes: set[float] = set()  # currently visible raw strikes
        self.all_strikes: list[float] = []      # all strikes before filtering

        # Theo computation cache — written from any thread, read by UI timer
        self._cached_theos: dict[float, tuple[float, float]] = {}  # raw_strike -> (bid_theo, ask_theo)
        self._theo_last_time: dict[float, float] = {}   # raw_strike -> last update timestamp
        self._theo_latency_ema: dict[float, float] = {}  # raw_strike -> EMA of latency in ms

        # Vol smile fitting state
        self._smile_coeffs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._smoothed_iv: dict[float, float] = {}      # disp_strike -> EWM smoothed IV
        self._last_smile_fit_time: float = 0.0           # monotonic time of last fit
        self._last_smile_spot: float = 0.0               # spot at last fit
        self._smile_span: int = 10                       # EWM span
        self._smile_otm_pct: float = 0.04                # fit on strikes within 4%
        self._smile_spot_threshold: float = 0.005        # refit on 0.5% spot move

        self._build_ui()
        self._apply_stylesheet()
        self._fetch_balance()

        # Timer — update table (non-theo columns) every 1s
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._update_table)

        # Fast timer — update theos + feed strategies every 200ms for all visible strikes
        self.fast_timer = QTimer()
        self.fast_timer.timeout.connect(self._update_fast_theos)
        self.fast_timer.start(200)

        # Balance refresh every 60s
        self.balance_timer = QTimer()
        self.balance_timer.timeout.connect(self._fetch_balance)
        self.balance_timer.start(60_000)

        # Position refresh every 10s
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._refresh_positions)
        self.position_timer.start(10_000)

        # Order audit every 30s
        self.audit_timer = QTimer()
        self.audit_timer.timeout.connect(self._audit_orders)
        self.audit_timer.start(30_000)

        # Event discovery refresh every 60s
        self.event_timer = QTimer()
        self.event_timer.timeout.connect(self._refresh_events)
        self.event_timer.start(60_000)

        # Vol smile refit every 60s
        self.smile_timer = QTimer()
        self.smile_timer.timeout.connect(self._refit_vol_smile)
        self.smile_timer.start(60_000)

        self.refresh_timer.start(1000)

        # Start
        self._on_series_changed(0)

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Top row: Series, Event, Spot ---
        top = QHBoxLayout()

        top.addWidget(QLabel("Crypto:"))
        self.series_combo = QComboBox()
        for s in CRYPTO_SERIES:
            self.series_combo.addItem(s["name"])
        self.series_combo.setMaximumWidth(120)
        self.series_combo.currentIndexChanged.connect(self._on_series_changed)
        top.addWidget(self.series_combo)

        top.addSpacing(15)
        top.addWidget(QLabel("Event:"))
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(280)
        self.event_combo.currentIndexChanged.connect(self._on_event_changed)
        top.addWidget(self.event_combo)

        top.addSpacing(15)
        top.addWidget(QLabel("Spot:"))
        self.spot_label = QLabel("--")
        self.spot_label.setFont(QFont("Courier", 14, QFont.Weight.Bold))
        self.spot_label.setStyleSheet("color: #facc15;")
        top.addWidget(self.spot_label)

        top.addSpacing(15)
        top.addWidget(QLabel("Balance:"))
        self.balance_label = QLabel("--")
        self.balance_label.setFont(QFont("Courier", 14, QFont.Weight.Bold))
        self.balance_label.setStyleSheet("color: #22c55e;")
        top.addWidget(self.balance_label)

        top.addWidget(QLabel("Portfolio:"))
        self.portfolio_label = QLabel("--")
        self.portfolio_label.setFont(QFont("Courier", 14, QFont.Weight.Bold))
        self.portfolio_label.setStyleSheet("color: #22c55e;")
        top.addWidget(self.portfolio_label)

        top.addSpacing(15)
        top.addWidget(QLabel("OTM%:"))
        self.otm_filter_input = QLineEdit(f"{self.otm_filter_pct}")
        self.otm_filter_input.setMaximumWidth(45)
        self.otm_filter_input.setStyleSheet(
            "QLineEdit{background:#141923;color:#c8cdd5;"
            "border:1px solid #1e2736;border-radius:3px;padding:2px 4px;}"
        )
        self.otm_filter_input.editingFinished.connect(self._on_otm_filter_changed)
        top.addWidget(self.otm_filter_input)

        top.addSpacing(15)
        top.addWidget(QLabel("Rate%:"))
        self.rate_input = QLineEdit("4.3")
        self.rate_input.setMaximumWidth(45)
        self.rate_input.setStyleSheet(
            "QLineEdit{background:#141923;color:#c8cdd5;"
            "border:1px solid #1e2736;border-radius:3px;padding:2px 4px;}"
        )
        self.rate_input.editingFinished.connect(self._on_rate_changed)
        top.addWidget(self.rate_input)

        top.addStretch()

        # Countdown labels
        self.kalshi_countdown = QLabel("")
        self.kalshi_countdown.setStyleSheet("color:#facc15;font-size:11px;")
        top.addWidget(self.kalshi_countdown)

        top.addSpacing(15)
        self.deribit_countdown = QLabel("")
        self.deribit_countdown.setStyleSheet("color:#5a6270;font-size:11px;")
        top.addWidget(self.deribit_countdown)

        top.addSpacing(15)
        self.iv_btn = QPushButton("IV Smile")
        self.iv_btn.setMaximumWidth(80)
        self.iv_btn.setStyleSheet(
            "QPushButton{background:#1e2736;color:#c8cdd5;border:1px solid #2d3a4d;"
            "border-radius:3px;padding:4px 8px;}"
            "QPushButton:hover{background:#2d3a4d;}"
        )
        self.iv_btn.clicked.connect(self._open_iv_window)
        top.addWidget(self.iv_btn)

        top.addSpacing(5)
        self.portfolio_btn = QPushButton("Portfolio")
        self.portfolio_btn.setMaximumWidth(80)
        self.portfolio_btn.setStyleSheet(
            "QPushButton{background:#1e2736;color:#c8cdd5;border:1px solid #2d3a4d;"
            "border-radius:3px;padding:4px 8px;}"
            "QPushButton:hover{background:#2d3a4d;}"
        )
        self.portfolio_btn.clicked.connect(self._open_portfolio_window)
        top.addWidget(self.portfolio_btn)

        layout.addLayout(top)

        # --- Second row: Auto Edge + Deribit status ---
        row2 = QHBoxLayout()

        _btn_style = ("QPushButton{background:#1e2736;color:#c8cdd5;border:1px solid #2d3a4d;"
                      "border-radius:3px;padding:4px 8px;}"
                      "QPushButton:hover{background:#2d3a4d;}"
                      "QPushButton:checked{background:#22c55e;color:#000;}")

        self.auto_edge_btn = QPushButton("Auto Edge")
        self.auto_edge_btn.setCheckable(True)
        self.auto_edge_btn.setChecked(False)
        self.auto_edge_btn.setMaximumWidth(80)
        self.auto_edge_btn.setStyleSheet(_btn_style)
        self.auto_edge_btn.toggled.connect(self._on_auto_edge_toggled)
        row2.addWidget(self.auto_edge_btn)

        self.auto_edge_config_btn = QPushButton("Config")
        self.auto_edge_config_btn.setMaximumWidth(55)
        self.auto_edge_config_btn.setStyleSheet(
            "QPushButton{background:#1e2736;color:#c8cdd5;border:1px solid #2d3a4d;"
            "border-radius:3px;padding:4px 8px;}"
            "QPushButton:hover{background:#2d3a4d;}"
        )
        self.auto_edge_config_btn.clicked.connect(self._open_auto_edge_config)
        row2.addWidget(self.auto_edge_config_btn)

        # Store auto edge params (load saved or use defaults)
        _defaults = {
            "base_bid": 1.0, "base_ask": 3.0,
            "atm_pct": 3.0, "scale": 0.5, "interval": 60,
        }
        saved_ae = _load_auto_edge_params()
        self._auto_edge_params = {k: saved_ae.get(k, v) for k, v in _defaults.items()}

        self._auto_edge_timer = QTimer()
        self._auto_edge_timer.timeout.connect(self._recompute_auto_edges)

        row2.addSpacing(15)
        row2.addWidget(QLabel("IV Source:"))
        self.iv_source_combo = QComboBox()
        self.iv_source_combo.addItems(["Auto", "Weekly"])
        self.iv_source_combo.setMaximumWidth(80)
        self.iv_source_combo.setStyleSheet(
            "QComboBox{background:#141923;color:#c8cdd5;"
            "border:1px solid #1e2736;border-radius:3px;padding:2px 4px;}"
        )
        self.iv_source_combo.currentTextChanged.connect(self._on_iv_source_changed)
        row2.addWidget(self.iv_source_combo)

        row2.addSpacing(10)
        self.deribit_label = QLabel("")
        self.deribit_label.setStyleSheet("color:#5a6270;font-size:11px;")
        row2.addWidget(self.deribit_label)

        row2.addSpacing(10)
        self.deribit_age_label = QLabel("")
        self.deribit_age_label.setStyleSheet("color:#5a6270;font-size:11px;")
        row2.addWidget(self.deribit_age_label)
        self._last_deribit_update = None

        row2.addSpacing(10)
        self.rate_limit_label = QLabel("")
        self.rate_limit_label.setStyleSheet("color:#5a6270;font-size:11px;")
        self.rate_limit_label.hide()
        row2.addWidget(self.rate_limit_label)

        row2.addStretch()

        layout.addLayout(row2)

        # --- Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(17)
        self.table.setHorizontalHeaderLabels([
            "Strike", "OTM%", "Yes Bid", "Yes Ask", "Theo", "Edge",
            "Deribit IV", "Smoothed IV", "Position", "Δ", "γ", "ν", "θ", "Order", "PnL", "", "",
        ])
        header = self.table.horizontalHeader()
        # All columns interactive (user-resizable), with sensible defaults
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 80)    # Strike
        self.table.setColumnWidth(1, 50)    # OTM%
        self.table.setColumnWidth(2, 90)    # Yes Bid
        self.table.setColumnWidth(3, 90)    # Yes Ask
        self.table.setColumnWidth(4, 130)   # Theo
        self.table.setColumnWidth(5, 75)    # Edge
        self.table.setColumnWidth(6, 120)   # Deribit IV
        self.table.setColumnWidth(7, 85)    # Smoothed IV
        self.table.setColumnWidth(8, 95)    # Position
        self.table.setColumnWidth(9, 85)    # Δ
        self.table.setColumnWidth(10, 85)   # γ
        self.table.setColumnWidth(11, 85)   # ν
        self.table.setColumnWidth(12, 85)   # θ
        self.table.setColumnWidth(13, 160)  # Order
        self.table.setColumnWidth(14, 90)   # PnL
        self.table.setColumnWidth(15, 50)   # ON/OFF button
        self.table.setColumnWidth(16, 50)   # FLAT button
        header.setStretchLastSection(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_right_click)
        self.table.cellDoubleClicked.connect(self._on_strike_clicked)
        layout.addWidget(self.table)

    # =========================================================================
    # Series & Event Selection
    # =========================================================================

    def _current_series_ticker(self) -> str:
        idx = self.series_combo.currentIndex()
        if 0 <= idx < len(CRYPTO_SERIES):
            return CRYPTO_SERIES[idx]["ticker"]
        return "KXBTCD"

    def _current_coinbase_product(self) -> str:
        idx = self.series_combo.currentIndex()
        if 0 <= idx < len(CRYPTO_SERIES):
            return CRYPTO_SERIES[idx].get("coinbase_product", "BTC-USD")
        return "BTC-USD"

    def _on_series_changed(self, index):
        """User picked a different crypto series."""
        # Stop old price feed
        if self.price_feed:
            self.price_feed.stop()
            self.price_feed = None

        # Stop old WS feeds
        if self.ws_feed:
            self.ws_feed.stop()
            self.ws_feed = None
        if self.deribit_ws:
            self.deribit_ws.stop()
            self.deribit_ws = None

        # Start new price feed
        product = self._current_coinbase_product()
        self.price_feed = CryptoPriceFeed(self._on_price, product)
        self.price_feed.start()

        # Discover events
        series = self._current_series_ticker()
        self._discover_worker = DiscoverWorker(self.api, series)
        self._discover_worker.finished.connect(self._on_events_discovered)
        self._discover_worker.error.connect(lambda e: print(f"[Discovery] Error: {e}"))
        self._discover_worker.start()

    def _refresh_events(self):
        """Periodically re-discover events to pick up new ones and drop expired."""
        series = self._current_series_ticker()
        self._discover_worker = DiscoverWorker(self.api, series)
        self._discover_worker.finished.connect(self._on_events_refreshed)
        self._discover_worker.error.connect(lambda e: print(f"[Discovery] Error: {e}"))
        self._discover_worker.start()

    def _on_events_discovered(self, events):
        """Initial discovery — populate combo and select first event."""
        self.events = self._filter_expired(events)
        self._rebuild_event_combo()
        if self.events:
            self._on_event_changed(0)

    def _on_events_refreshed(self, events):
        """Periodic refresh — update list, keep current selection if still valid."""
        current_et = None
        if self.current_event:
            current_et = self.current_event.get("event_ticker")

        self.events = self._filter_expired(events)
        self._rebuild_event_combo()

        # Try to re-select the same event
        if current_et:
            for i, ev in enumerate(self.events):
                if ev["event_ticker"] == current_et:
                    self.event_combo.blockSignals(True)
                    self.event_combo.setCurrentIndex(i)
                    self.event_combo.blockSignals(False)
                    return
        # Current event expired or gone — select first
        if self.events:
            self._on_event_changed(0)

    @staticmethod
    def _filter_expired(events: list) -> list:
        """Remove events that closed more than 10 minutes ago."""
        now = datetime.now(tz=timezone.utc)
        filtered = []
        for ev in events:
            close_str = ev.get("close_time", "")
            if not close_str:
                filtered.append(ev)
                continue
            try:
                close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                if (now - close_utc).total_seconds() < 600:  # keep for 10 min after close
                    filtered.append(ev)
            except Exception:
                filtered.append(ev)
        return filtered

    def _rebuild_event_combo(self):
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        for ev in self.events:
            n = ev["num_brackets"]
            self.event_combo.addItem(f"{ev['event_ticker']} ({n} markets)")
        self.event_combo.blockSignals(False)

    def _on_event_changed(self, index):
        if index < 0 or index >= len(self.events):
            return

        # Stash current strategies + market_data so they keep running
        # in the background (orders stay live, positions stay tracked).
        if self.current_event and self.strategies:
            old_et = self.current_event.get("event_ticker", "")
            if old_et:
                self._stashed_strategies[old_et] = dict(self.strategies)
                self._stashed_market_data[old_et] = dict(self.market_data)
                self._stashed_events[old_et] = self.current_event
        self.strategies.clear()

        self.current_event = self.events[index]
        self._smoothed_iv.clear()
        self._last_smile_spot = 0.0
        new_et = self.current_event.get("event_ticker", "")
        markets = self.current_event["markets"]

        # Check if we have stashed strategies for this event
        if new_et in self._stashed_strategies:
            self.strategies = self._stashed_strategies.pop(new_et)
            self.market_data = self._stashed_market_data.pop(new_et, {})
            self._stashed_events.pop(new_et, None)
            # Fill in any new markets not in the stash
            for m in markets:
                raw = parse_strike(m["ticker"])
                if raw > 0 and raw not in self.market_data:
                    self.market_data[raw] = {
                        "ticker": m["ticker"],
                        "yes_bid": 0.0, "yes_ask": 0.0,
                        "bid_size": 0, "ask_size": 0,
                        "position": 0, "exposure": 0.0,
                        "realized_pnl": 0.0,
                    }
        else:
            # Build fresh market_data
            self.market_data = {}
            for m in markets:
                raw = parse_strike(m["ticker"])
                if raw > 0:
                    self.market_data[raw] = {
                        "ticker": m["ticker"],
                        "yes_bid": 0.0, "yes_ask": 0.0,
                        "bid_size": 0, "ask_size": 0,
                        "position": 0, "exposure": 0.0,
                        "realized_pnl": 0.0,
                    }

        raw_strikes = set(self.market_data.keys())
        self.all_strikes = sorted(raw_strikes)
        self.all_display_strikes = [display_strike(s) for s in self.all_strikes]
        self.visible_strikes = set(self.all_strikes)  # start with all visible
        self._theo_last_time.clear()
        self._theo_latency_ema.clear()

        self._apply_otm_filter()
        if new_et not in self._stashed_strategies:
            # Only restore from exchange if we didn't unstash
            self._restore_strategies()
        self._start_ws_feed()
        self._fetch_deribit()

    # =========================================================================
    # OTM% Filter
    # =========================================================================

    def _on_otm_filter_changed(self):
        """User changed the OTM% filter input."""
        try:
            val = float(self.otm_filter_input.text())
            if val > 0:
                self.otm_filter_pct = val
                self._apply_otm_filter()
        except ValueError:
            pass

    def _on_rate_changed(self):
        """User changed the risk-free rate input."""
        try:
            val = float(self.rate_input.text())
            self.pricer.risk_free_rate = val / 100.0
        except ValueError:
            pass

    def _open_auto_edge_config(self):
        """Open dialog to configure auto edge parameters."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Auto Edge Configuration")
        dlg.setStyleSheet(
            "QDialog{background:#0d1117;color:#c8cdd5;}"
            "QLabel{color:#c8cdd5;font-size:12px;}"
            "QLineEdit{background:#161b22;color:#e6edf3;border:1px solid #30363d;"
            "border-radius:3px;padding:4px 6px;font-size:12px;}"
        )
        layout = QFormLayout(dlg)

        p = self._auto_edge_params
        inputs = {}
        fields = [
            ("base_bid", "Base Bid Edge (¢)", p["base_bid"]),
            ("base_ask", "Base Ask Edge (¢)", p["base_ask"]),
            ("atm_pct", "ATM Threshold (%)", p["atm_pct"]),
            ("scale", "Scale (¢ per % beyond ATM)", p["scale"]),
            ("interval", "Recompute Interval (s)", int(p["interval"])),
        ]
        for key, label, val in fields:
            inp = QLineEdit(str(val))
            inp.setFixedWidth(100)
            inputs[key] = inp
            layout.addRow(QLabel(label), inp)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet(
            "QPushButton{background:#238636;color:#fff;border:none;"
            "border-radius:3px;padding:6px 16px;}"
            "QPushButton:hover{background:#2ea043;}"
        )
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton{background:#21262d;color:#c8cdd5;border:1px solid #30363d;"
            "border-radius:3px;padding:6px 16px;}"
            "QPushButton:hover{background:#30363d;}"
        )
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

        def accept():
            try:
                self._auto_edge_params["base_bid"] = float(inputs["base_bid"].text())
                self._auto_edge_params["base_ask"] = float(inputs["base_ask"].text())
                self._auto_edge_params["atm_pct"] = float(inputs["atm_pct"].text())
                self._auto_edge_params["scale"] = float(inputs["scale"].text())
                self._auto_edge_params["interval"] = max(5, int(float(inputs["interval"].text())))
            except ValueError:
                QMessageBox.warning(dlg, "Invalid Input", "All fields must be numeric.")
                return
            # Update timer interval if auto edge is active
            if self.auto_edge_btn.isChecked():
                self._auto_edge_timer.start(self._auto_edge_params["interval"] * 1000)
            _save_auto_edge_params(self._auto_edge_params)
            # Always recompute immediately on config change
            self._recompute_auto_edges(force=True)
            dlg.accept()

        ok_btn.clicked.connect(accept)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    def _on_auto_edge_toggled(self, checked: bool):
        """Toggle between manual and auto edge mode."""
        if checked:
            interval = self._auto_edge_params["interval"]
            self._auto_edge_timer.start(max(interval, 5) * 1000)
            self._recompute_auto_edges()  # run immediately
            print(f"[App] Auto edge ON (interval={interval}s)")
        else:
            self._auto_edge_timer.stop()
            print("[App] Auto edge OFF (manual mode)")

    def _recompute_auto_edges(self, force=False):
        """Recompute edges for all active strategies based on OTM%."""
        if not force and not self.auto_edge_btn.isChecked():
            return

        p = self._auto_edge_params
        base_bid = p["base_bid"] / 100.0   # cents to dollars
        base_ask = p["base_ask"] / 100.0
        atm_pct = p["atm_pct"]
        scale_per_pct = p["scale"] / 100.0  # cents to dollars

        spot = self.spot_price
        if spot <= 0:
            return

        updated = 0
        for ticker, strat in self.strategies.items():
            otm = abs(strat.strike - spot) / spot * 100
            beyond = max(otm - atm_pct, 0)
            new_eb = round(base_bid + beyond * scale_per_pct, 4)
            new_ea = round(base_ask + beyond * scale_per_pct, 4)

            if abs(strat.edge_bid - new_eb) > 0.0001 or abs(strat.edge_ask - new_ea) > 0.0001:
                strat.edge_bid = new_eb
                strat.edge_ask = new_ea
                updated += 1

        if updated > 0:
            print(f"[App] Auto edge: updated {updated} strategies "
                  f"(base={base_bid*100:.1f}¢/{base_ask*100:.1f}¢, "
                  f"atm={atm_pct}%, scale={scale_per_pct*100:.1f}¢/% beyond)")

    def _apply_otm_filter(self):
        """Filter strikes by OTM% with hysteresis, rebuild table.

        A strike enters the visible set when |OTM%| <= otm_filter_pct.
        It stays visible until |OTM%| > otm_filter_pct + hysteresis.
        Strikes with active strategies are always shown.
        """
        if self.spot_price <= 0:
            # No spot yet — show all
            self.strikes = list(self.all_strikes)
            self.display_strikes = list(self.all_display_strikes)
            self._rebuild_table()
            return

        new_visible = set()
        threshold = self.otm_filter_pct
        hyst = self.otm_hysteresis

        for raw, disp in zip(self.all_strikes, self.all_display_strikes):
            otm = abs((disp - self.spot_price) / self.spot_price * 100)

            # Always show strikes with active strategies or positions
            strat = self.strategies.get(raw)
            if strat and (strat.active or strat.position != 0):
                new_visible.add(raw)
                continue
            data = self.market_data.get(raw, {})
            if data.get("position", 0) != 0:
                new_visible.add(raw)
                continue

            currently_visible = raw in self.visible_strikes
            if currently_visible:
                # Keep until exceeds threshold + hysteresis
                if otm <= threshold + hyst:
                    new_visible.add(raw)
            else:
                # Add when within threshold
                if otm <= threshold:
                    new_visible.add(raw)

        self.visible_strikes = new_visible
        self.strikes = [s for s in self.all_strikes if s in new_visible]
        self.display_strikes = [display_strike(s) for s in self.strikes]
        self._rebuild_table()

    # =========================================================================
    # Kalshi WS Feed
    # =========================================================================

    def _start_ws_feed(self):
        """Start websocket feed for current event + any stashed events with
        active strategies, so fills on background orders are still captured."""
        if self.ws_feed:
            self.ws_feed.stop()
            self.ws_feed = None

        tickers = [d["ticker"] for d in self.market_data.values()]

        # Also subscribe to tickers from stashed events that have strategies
        for et, stashed_md in self._stashed_market_data.items():
            stashed_strats = self._stashed_strategies.get(et, {})
            for raw, data in stashed_md.items():
                if raw in stashed_strats:
                    t = data["ticker"]
                    if t not in tickers:
                        tickers.append(t)

        if not tickers:
            return

        self.ws_feed = KalshiWsFeed(self.api, self._on_ws_update,
                                     on_fill=self._on_ws_fill)
        self.ws_feed.start(tickers)

    def _find_market_data_for_ticker(self, ticker: str):
        """Find the (market_data_dict, strategies_dict, raw_strike) for a ticker.
        Checks current event first, then stashed events."""
        for raw, data in self.market_data.items():
            if data["ticker"] == ticker:
                return data, self.strategies, raw
        for et, smd in self._stashed_market_data.items():
            for raw, data in smd.items():
                if data["ticker"] == ticker:
                    return data, self._stashed_strategies.get(et, {}), raw
        return None, None, None

    def _on_ws_update(self, ticker: str, yes_bid: float, yes_ask: float,
                      bid_size: int = 0, ask_size: int = 0):
        """Callback from WS feed — update market_data (current or stashed)."""
        data, _, _ = self._find_market_data_for_ticker(ticker)
        if data:
            data["yes_bid"] = yes_bid
            data["yes_ask"] = yes_ask
            data["bid_size"] = bid_size
            data["ask_size"] = ask_size

    def _on_ws_fill(self, ticker: str, action: str, side: str,
                    price: float, count: int):
        """Callback from WS feed — update position on market_data.
        Works for both current and stashed events.
        PnL is recomputed from all fills by _refresh_positions (REST)."""
        data, strats, raw = self._find_market_data_for_ticker(ticker)
        if data is None:
            return

        pos = data.get("position", 0)
        if side == "yes":
            pos += count if action == "buy" else -count
        elif side == "no":
            pos += count if action == "sell" else -count
        data["position"] = pos

        # Sync to strategy — clear resting order so it reposts
        strat = strats.get(raw) if strats else None
        if strat:
            strat.position = pos
            strat.on_fill()

        # Flash the position cell
        if raw is not None:
            self._fill_flash[raw] = time.monotonic()

        # Trigger immediate REST refresh to recompute PnL from all fills
        self._refresh_positions()

        # Play fill sound
        try:
            import subprocess
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        print(f"[App Fill] {ticker} {action} {side} x{count} "
              f"@ ${price:.2f}  pos={pos}")

    # =========================================================================
    # Price Feed
    # =========================================================================

    def _on_price(self, price: float, bid: float = 0.0, ask: float = 0.0):
        self.spot_price = price
        if bid > 0:
            self.spot_bid = bid
        if ask > 0:
            self.spot_ask = ask
        # Compute theos + feed strategies immediately (runs on WS thread)
        self._recompute_and_trade()
        self._ui_dirty = True  # signal UI timer to refresh display

        # Refit vol smile on big spot move
        if self._last_smile_spot > 0:
            move = abs(price - self._last_smile_spot) / self._last_smile_spot
            if move >= self._smile_spot_threshold:
                self._refit_vol_smile()

    # =========================================================================
    # Balance
    # =========================================================================

    def _fetch_balance(self):
        try:
            data = self.api.get_balance()
            if not hasattr(self, '_balance_logged'):
                print(f"[App] Balance response: {data}")
                self._balance_logged = True
            self._balance_cents = data.get("balance", 0)
            portfolio_value_cents = data.get("portfolio_value", 0)
            balance_dollars = self._balance_cents / 100
            portfolio_dollars = (self._balance_cents + portfolio_value_cents) / 100
            self._portfolio_history.append((datetime.now(tz=timezone.utc), portfolio_dollars))
            self.balance_label.setText(f"${balance_dollars:,.2f}")

            # Track portfolio baseline for session change %
            if not hasattr(self, '_portfolio_baseline'):
                self._portfolio_baseline = portfolio_dollars
            change = portfolio_dollars - self._portfolio_baseline
            if self._portfolio_baseline > 0:
                pct = (change / self._portfolio_baseline) * 100
                self.portfolio_label.setText(
                    f"${portfolio_dollars:,.2f} ({pct:+.2f}%)")
            else:
                self.portfolio_label.setText(f"${portfolio_dollars:,.2f}")
        except Exception:
            pass

    # =========================================================================
    # Rate Limit
    # =========================================================================

    def _on_rate_limit(self, remaining, limit, reset_ts, endpoint=""):
        """Called from API on every response with rate limit headers."""
        self._rate_limit_remaining = remaining
        self._rate_limit_total = limit
        if remaining == 0:
            self._rate_limit_hit_time = time.time()
            self._rate_limit_endpoint = endpoint

    def _update_rate_limit_label(self):
        """Update the rate limit warning label. Called from table tick."""
        # Show if we got a 429 in the last 10 seconds
        if self._rate_limit_hit_time > 0:
            elapsed = time.time() - self._rate_limit_hit_time
            if elapsed < 10:
                ep = getattr(self, "_rate_limit_endpoint", "")
                txt = f"RATE LIMITED ({ep})" if ep else "RATE LIMITED"
                self.rate_limit_label.setText(txt)
                self.rate_limit_label.setStyleSheet(
                    "color:#ef4444;font-size:12px;font-weight:bold;")
                self.rate_limit_label.show()
                return
            else:
                self._rate_limit_hit_time = 0

        self.rate_limit_label.hide()

    # =========================================================================
    # Deribit
    # =========================================================================

    def _on_iv_source_changed(self, text: str):
        """User changed IV source dropdown."""
        print(f"[App] IV source changed to: {text}")
        if text == "Weekly" and not self.weekly_deribit_ws:
            self._fetch_weekly_deribit()
        self._update_deribit_status_label()
        self._recompute_and_trade()
        self._ui_dirty = True

    def _fetch_deribit(self):
        """Discover Deribit instruments, then start WS feed."""
        if not self.current_event:
            return

        # Stop any existing Deribit WS feed
        if self.deribit_ws:
            self.deribit_ws.stop()
            self.deribit_ws = None

        series = self._current_series_ticker()
        currency = KALSHI_TO_DERIBIT_CURRENCY.get(series)
        if not currency:
            self.deribit_label.setText(f"No Deribit for {series}")
            return

        self.pricer.currency = currency
        self.weekly_pricer.currency = currency

        close_time = self.current_event.get("close_time", "")
        expiry_str = find_deribit_expiry(close_time, currency=currency)
        if not expiry_str:
            self.deribit_label.setText("No expiry match")
            return

        if self._deribit_discover_worker and self._deribit_discover_worker.isRunning():
            return

        self.deribit_label.setText(f"Discovering {currency} {expiry_str}...")
        self._deribit_discover_worker = DeribitDiscoverWorker(self.pricer, expiry_str)
        self._deribit_discover_worker.finished.connect(self._on_instruments_discovered)
        self._deribit_discover_worker.error.connect(
            lambda e: self.deribit_label.setText(f"Error: {e}")
        )
        self._deribit_discover_worker.start()

        # Also start the weekly feed if IV source is set to Weekly
        if self.iv_source_combo.currentText() == "Weekly":
            self._fetch_weekly_deribit()

    def _fetch_weekly_deribit(self):
        """Discover and connect to the weekly (nearest Friday) Deribit expiry."""
        series = self._current_series_ticker()
        currency = KALSHI_TO_DERIBIT_CURRENCY.get(series)
        if not currency:
            return

        if self._weekly_discover_worker and self._weekly_discover_worker.isRunning():
            return

        # Find all available expiries and pick the nearest weekly (Friday)
        try:
            available = list_deribit_expiries(currency)
        except Exception as e:
            print(f"[App] Failed to list Deribit expiries: {e}")
            return

        from datetime import timedelta
        _months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

        now = datetime.now(tz=timezone.utc).date()
        best = None
        best_date = None
        for exp_str in available:
            try:
                for m_name, m_num in _months.items():
                    idx = exp_str.find(m_name)
                    if idx > 0:
                        d = int(exp_str[:idx])
                        y = 2000 + int(exp_str[idx + 3:])
                        exp_date = datetime(y, m_num, d).date()
                        # Weekly = closest upcoming Friday (weekday 4)
                        if exp_date.weekday() != 4:
                            break
                        if exp_date >= now:
                            if best_date is None or exp_date < best_date:
                                best = exp_str
                                best_date = exp_date
                        break
            except Exception:
                continue

        if not best:
            print("[App] No weekly Deribit expiry found")
            return

        # Don't restart if already connected to this expiry
        if best == self._weekly_expiry_str and self.weekly_deribit_ws:
            return

        self._weekly_expiry_str = best
        print(f"[App] Discovering weekly Deribit expiry: {best}")

        if self.weekly_deribit_ws:
            self.weekly_deribit_ws.stop()
            self.weekly_deribit_ws = None

        self._weekly_discover_worker = DeribitDiscoverWorker(self.weekly_pricer, best)
        self._weekly_discover_worker.finished.connect(self._on_weekly_instruments_discovered)
        self._weekly_discover_worker.error.connect(
            lambda e: print(f"[App] Weekly Deribit error: {e}")
        )
        self._weekly_discover_worker.start()

    def _on_weekly_instruments_discovered(self, expiry_str: str, instruments: list):
        """Weekly instruments found — start WS feed."""
        print(f"[App] Weekly Deribit {expiry_str}: {len(instruments)} options")
        self.weekly_deribit_ws = DeribitWsFeed(
            self.weekly_pricer, self._on_weekly_deribit_update
        )
        self.weekly_deribit_ws.start(instruments)

    def _on_weekly_deribit_update(self):
        """Weekly Deribit data refreshed — recompute if using weekly IVs."""
        if self.iv_source_combo.currentText() == "Weekly":
            self._last_deribit_update = time.monotonic()
            self._recompute_and_trade()
            self._ui_dirty = True
            self._update_deribit_status_label()

    def _on_instruments_discovered(self, expiry_str: str, instruments: list):
        """Instruments found via REST — now start WS feed for live prices."""
        self.deribit_label.setText(
            f"Connecting WS for {len(instruments)} options..."
        )
        self.deribit_ws = DeribitWsFeed(self.pricer, self._on_deribit_update)
        self.deribit_ws.start(instruments)

    @staticmethod
    def _format_countdown(seconds: float) -> str:
        """Format seconds remaining as Xd Xh Xm Xs."""
        if seconds <= 0:
            return "EXPIRED"
        s = int(seconds)
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        parts = []
        if d > 0:
            parts.append(f"{d}d")
        if h > 0 or d > 0:
            parts.append(f"{h}h")
        parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)

    def _update_countdowns(self):
        """Update Kalshi and Deribit expiry countdown labels."""
        now = datetime.now(tz=timezone.utc)

        # Kalshi event expiry
        if self.current_event:
            close_str = self.current_event.get("close_time", "")
            if close_str:
                try:
                    close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    remaining = (close_utc - now).total_seconds()
                    t_years = remaining / (365.25 * 86400)
                    self.kalshi_countdown.setText(
                        f"Kalshi: {self._format_countdown(remaining)}  T={t_years:.6f}y"
                    )
                except Exception:
                    self.kalshi_countdown.setText("")
            else:
                self.kalshi_countdown.setText("")
        else:
            self.kalshi_countdown.setText("")

        # Deribit expiry
        ts_ms = self.pricer.expiration_ts_ms
        if ts_ms:
            exp_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            remaining = (exp_utc - now).total_seconds()
            self.deribit_countdown.setText(
                f"Deribit: {self._format_countdown(remaining)}"
            )
        else:
            self.deribit_countdown.setText("")

    def _on_deribit_update(self):
        """Called by DeribitWsFeed after each density rebuild."""
        self._last_deribit_update = time.monotonic()
        self._recompute_and_trade()
        self._ui_dirty = True
        self._update_deribit_status_label()

    def _update_deribit_status_label(self):
        """Update the deribit info label based on active IV source."""
        p = self._get_iv_pricer()
        if p.ready:
            lo, hi = p.strike_range
            source = "weekly" if p is self.weekly_pricer else "daily"
            self.deribit_label.setText(
                f"{p.expiry_str}  {p.n_options} opts  "
                f"[${lo:,.0f} - ${hi:,.0f}]  ({source}, live)"
            )
        else:
            self.deribit_label.setText("Density failed")

    def _find_deribit_strikes(self, K: float) -> tuple:
        """Find the two Deribit option strikes straddling K.

        Returns (k_low, k_high, dc_dk) or (None, None, None) if not found.
        """
        if not self.pricer.ready or not self.pricer.options:
            return None, None, None

        opts = self.pricer.options
        for i in range(len(opts) - 1):
            if opts[i].strike <= K <= opts[i + 1].strike:
                k_lo = opts[i].strike
                k_hi = opts[i + 1].strike
                h = k_hi - k_lo
                if h <= 0:
                    continue
                dc_dk = (opts[i + 1].call_price_usd - opts[i].call_price_usd) / h
                return k_lo, k_hi, dc_dk

        return None, None, None

    def _find_closest_iv(self, K: float) -> tuple[float | None, float | None, float | None]:
        """Find bid_iv and ask_iv (as %) from the closest Deribit option to K.

        Returns (bid_iv_pct, ask_iv_pct, strike) or (None, None, None).
        """
        iv_p = self._get_iv_pricer()
        if not iv_p.options:
            return None, None, None
        best_opt = None
        best_dist = float("inf")
        for opt in iv_p.options:
            if opt.bid_iv > 0 or opt.ask_iv > 0:
                dist = abs(opt.strike - K)
                if dist < best_dist:
                    best_dist = dist
                    best_opt = opt
        if best_opt is None:
            return None, None, None
        bid_pct = best_opt.bid_iv * 100.0 if best_opt.bid_iv > 0 else None
        ask_pct = best_opt.ask_iv * 100.0 if best_opt.ask_iv > 0 else None
        return bid_pct, ask_pct, best_opt.strike

    # =========================================================================
    # IV Smile Window
    # =========================================================================

    def _open_iv_window(self):
        """Open (or bring to front) the IV smile graph window."""
        if self.iv_window is None or not self.iv_window.isVisible():
            self.iv_window = IVSmileWindow(self._get_iv_pricer(), self)
        self.iv_window.show()
        self.iv_window.raise_()
        self.iv_window.update_chart()

    def _open_portfolio_window(self):
        """Open a window showing historical portfolio value (balance + positions)."""
        import zoneinfo
        ct = zoneinfo.ZoneInfo("America/Chicago")
        start_date = datetime(2026, 4, 7, tzinfo=timezone.utc)

        try:
            bal_data = self.api.get_balance()
            current_balance = bal_data.get("balance", 0) / 100
            current_portfolio_value = bal_data.get("portfolio_value", 0) / 100
            current_total = current_balance + current_portfolio_value
        except Exception as e:
            print(f"[App] Failed to fetch balance for portfolio: {e}")
            return

        try:
            all_fills = self.api.get_fills()
        except Exception as e:
            print(f"[App] Failed to fetch fills for portfolio: {e}")
            return

        if not all_fills:
            return

        # Sort fills by time
        fills_sorted = sorted(all_fills, key=lambda f: f.get("created_time", ""))

        # Compute cumulative PnL using FIFO and track net cash flow from trades
        # Portfolio value = starting_capital + cumulative_pnl + unrealized_value
        # We reconstruct: starting_capital = current_total - total_pnl_since_start
        run_pos = {}
        avg_entry = {}
        cumulative_pnl = 0.0
        net_cost = 0.0  # net cash spent on open positions

        # Track daily: {date_str: (cumulative_pnl, net_cost_of_open_positions)}
        daily_snapshot = {}

        for f in fills_sorted:
            t = f.get("ticker", "")
            created = f.get("created_time", "")
            if not created:
                continue
            try:
                fill_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                continue
            if fill_dt < start_date:
                continue

            action = f.get("action", "")
            count = int(float(f.get("count_fp", 0) or 0))
            yes_price = float(f.get("yes_price_dollars", 0) or 0)
            if count == 0 or yes_price == 0:
                continue

            if t not in run_pos:
                run_pos[t] = 0
                avg_entry[t] = 0.0

            prev = run_pos[t]
            delta = count if action == "buy" else -count
            new = prev + delta

            if prev == 0:
                avg_entry[t] = yes_price
            elif (prev > 0 and delta > 0) or (prev < 0 and delta < 0):
                total_cost = avg_entry[t] * abs(prev) + yes_price * abs(delta)
                avg_entry[t] = total_cost / abs(new) if new != 0 else 0
            else:
                closed = min(abs(delta), abs(prev))
                if prev > 0:
                    cumulative_pnl += (yes_price - avg_entry[t]) * closed
                else:
                    cumulative_pnl += (avg_entry[t] - yes_price) * closed
                if new == 0:
                    avg_entry[t] = 0.0
                elif (prev > 0 and new < 0) or (prev < 0 and new > 0):
                    avg_entry[t] = yes_price

            run_pos[t] = new

            day_str = fill_dt.astimezone(ct).strftime("%Y-%m-%d")
            daily_snapshot[day_str] = cumulative_pnl

        if not daily_snapshot:
            return

        # starting_capital = current_total - total cumulative PnL (realized) - unrealized
        # Since we don't know historical unrealized, approximate:
        # portfolio_value_on_day ≈ starting_capital + pnl_as_of_day
        total_realized = cumulative_pnl
        starting_capital = current_total - total_realized - current_portfolio_value

        dates = sorted(daily_snapshot.keys())
        values = [starting_capital + daily_snapshot[d] for d in dates]
        date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

        # Add today's actual value as the last point
        today_str = datetime.now(tz=ct).strftime("%Y-%m-%d")
        if not dates or dates[-1] != today_str:
            date_objs.append(datetime.strptime(today_str, "%Y-%m-%d"))
            values.append(current_total)

        win = QMainWindow(self)
        win.setWindowTitle("Portfolio Value")
        win.resize(700, 400)
        win.setStyleSheet("background:#0d1117;")

        fig = Figure(figsize=(7, 4), facecolor="#0d1117")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#141923")

        ax.scatter(date_objs, values, c="#3b82f6", s=50, zorder=5)
        ax.plot(date_objs, values, color="#5a6270", linewidth=0.8, alpha=0.5, zorder=1)

        ax.set_title(f"Portfolio: ${values[-1]:,.2f}" if values else "No data",
                     color="#c8cdd5", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", color="#5a6270", fontsize=10)
        ax.set_xlabel("Date (CT)", color="#5a6270", fontsize=10)
        ax.tick_params(colors="#5a6270", labelsize=9)
        ax.grid(True, color="#1e2736", alpha=0.5)

        for spine in ax.spines.values():
            spine.set_color("#1e2736")

        fig.autofmt_xdate()
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        win.setCentralWidget(canvas)
        win.show()
        self._portfolio_window = win

    # =========================================================================
    # Table
    # =========================================================================

    def _rebuild_table(self):
        self.table.setRowCount(len(self.strikes))
        for row, (raw, disp) in enumerate(zip(self.strikes, self.display_strikes)):
            self.table.setItem(row, 0, QTableWidgetItem(f"${disp:,.0f}"))
            for col in range(1, 15):
                self.table.setItem(row, col, QTableWidgetItem("--"))

            # Toggle button (col 15)
            strat = self.strategies.get(raw)
            btn = QPushButton("ON" if strat and strat.active else "OFF")
            btn.setStyleSheet(
                ("QPushButton{background:#1e2736;color:#22c55e;border:1px solid #22c55e;"
                 "border-radius:3px;padding:2px 6px;font-size:11px;}"
                 "QPushButton:hover{background:#2d3a4d;}")
                if strat and strat.active else
                ("QPushButton{background:#1e2736;color:#ef4444;border:1px solid #2d3a4d;"
                 "border-radius:3px;padding:2px 6px;font-size:11px;}"
                 "QPushButton:hover{background:#2d3a4d;}")
            )
            btn.clicked.connect(lambda checked, rs=raw: self._toggle_strategy(rs))
            self.table.setCellWidget(row, 15, btn)

            # Flatten button (col 16)
            flat_btn = QPushButton("FLAT")
            flat_btn.setStyleSheet(
                "QPushButton{background:#1e2736;color:#facc15;border:1px solid #2d3a4d;"
                "border-radius:3px;padding:2px 6px;font-size:11px;}"
                "QPushButton:hover{background:#2d3a4d;border-color:#facc15;}"
            )
            flat_btn.clicked.connect(lambda checked, rs=raw: self._flatten_position(rs))
            self.table.setCellWidget(row, 16, flat_btn)

    _filter_tick = 0

    def _update_table(self):
        self._update_countdowns()
        self._update_rate_limit_label()
        # Update Deribit age
        if self._last_deribit_update is not None:
            age = time.monotonic() - self._last_deribit_update
            if age < 60:
                self.deribit_age_label.setText(f"IV: {age:.0f}s ago")
                self.deribit_age_label.setStyleSheet(
                    "color:#22c55e;font-size:11px;" if age < 10 else "color:#5a6270;font-size:11px;")
            else:
                self.deribit_age_label.setText(f"IV: {age/60:.1f}m ago")
                self.deribit_age_label.setStyleSheet("color:#ef4444;font-size:11px;")
        # Update spot display
        if self.spot_price > 0:
            self.spot_label.setText(f"${self.spot_price:,.2f}")

        # Re-apply OTM filter every 3s as spot moves
        self._filter_tick += 1
        if self._filter_tick >= 3:
            self._filter_tick = 0
            old_set = set(self.strikes)
            self._apply_otm_filter()
            if set(self.strikes) != old_set:
                return  # table was rebuilt, skip rest of this tick

        for row, raw_strike in enumerate(self.strikes):
            if row >= self.table.rowCount():
                break

            disp = self.display_strikes[row]
            data = self.market_data.get(raw_strike, {})

            bid = data.get("yes_bid", 0.0)
            bid_size = data.get("bid_size", 0)
            ask = data.get("yes_ask", 0.0)
            ask_size = data.get("ask_size", 0)

            # OTM% — col 1
            item = self.table.item(row, 1)
            if item:
                if self.spot_price > 0:
                    otm_pct = (disp - self.spot_price) / self.spot_price * 100
                    item.setText(f"{otm_pct:+.1f}%")
                    if abs(otm_pct) >= 3:
                        item.setForeground(QColor("#facc15"))  # yellow = tail
                    elif abs(otm_pct) >= 1:
                        item.setForeground(QColor("#5a6270"))
                    else:
                        item.setForeground(QColor("#ef4444"))  # red = near ATM
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Yes Bid (size) — col 2 (green, cyan if we are BBO)
            # Yes Ask (size) — col 3 (red, cyan if we are BBO)
            # Both show implied vol as small sub-label
            strat = self.strategies.get(raw_strike)
            my_buy = strat.current_buy_price if strat else None
            my_sell = strat.current_sell_price if strat else None

            # Compute T for implied vol
            t_years = 0.0
            if self.current_event:
                close_iso = self.current_event.get("close_time", "")
                if close_iso:
                    try:
                        close_utc = datetime.fromisoformat(
                            close_iso.replace("Z", "+00:00"))
                        t_years = max(
                            (close_utc - datetime.now(tz=timezone.utc)).total_seconds()
                            / (365.25 * 86400), 0)
                    except Exception:
                        pass

            self._display_market_price(
                row, 2, bid, bid_size, disp, t_years,
                base_color="#22c55e",
                highlight_price=my_buy)
            self._display_market_price(
                row, 3, ask, ask_size, disp, t_years,
                base_color="#ef4444",
                highlight_price=my_sell)

            # Theo — col 4: handled by fast timer, skip here

            # Edge — col 5 (show bid/ask edges)
            strat = self.strategies.get(raw_strike)
            item = self.table.item(row, 5)
            if item:
                if strat:
                    item.setText(f"${strat.edge_bid:.2f}/${strat.edge_ask:.2f}")
                    item.setForeground(QColor("#c8cdd5"))
                elif self.auto_edge_btn.isChecked() and self.spot_price > 0:
                    # Preview auto edge for strikes without a strategy
                    p = self._auto_edge_params
                    base_bid = p["base_bid"] / 100.0
                    base_ask = p["base_ask"] / 100.0
                    atm_pct = p["atm_pct"]
                    scale_per_pct = p["scale"] / 100.0
                    otm = abs(disp - self.spot_price) / self.spot_price * 100
                    beyond = max(otm - atm_pct, 0)
                    eb = base_bid + beyond * scale_per_pct
                    ea = base_ask + beyond * scale_per_pct
                    item.setText(f"${eb:.2f}/${ea:.2f}")
                    item.setForeground(QColor("#5a6270"))  # dimmed to indicate preview
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            # Deribit IV — col 6 (bid / ask + strike used)
            self._display_iv(row, disp)

            # Smoothed IV — col 7
            self._display_smoothed_iv(row, disp)

            # Position — col 8 (qty + avg fill price)
            item = self.table.item(row, 8)
            if item:
                qty = strat.position if strat else data.get("position", 0)
                avg_px = data.get("avg_price", 0)
                if qty != 0:
                    if avg_px > 0:
                        item.setText(f"{qty} @ ${avg_px:.2f}")
                    else:
                        item.setText(f"{qty}")
                    color = "#ef4444" if qty < 0 else "#22c55e"
                    item.setForeground(QColor(color))
                else:
                    item.setText("0")
                    item.setForeground(QColor("#5a6270"))

                # Flash on recent fill
                flash_ts = self._fill_flash.get(raw_strike)
                if flash_ts and (time.monotonic() - flash_ts) < 1.5:
                    item.setBackground(QColor("#facc15"))
                    item.setForeground(QColor("#000000"))
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
                    if flash_ts:
                        del self._fill_flash[raw_strike]

            # Delta — col 9: position delta (dP/dSpot * position)
            self._display_greeks(row, raw_strike, disp, data)

            # Order — col 13 (show both sides, highlight flatten orders)
            self._display_orders(row, strat, raw_strike)

            # PnL — col 14: total PnL (session PnL)
            item = self.table.item(row, 14)
            if item:
                realized = strat.realized_pnl if strat else data.get("realized_pnl", 0)
                pos = strat.position if strat else data.get("position", 0)

                if pos != 0:
                    yes_bid = data.get("yes_bid", 0)
                    yes_ask = data.get("yes_ask", 0)
                    avg_px = data.get("avg_price", 0)
                    if pos > 0:
                        # Long: unrealized = (mark - entry) * qty
                        unrealized = (yes_bid - avg_px) * pos if avg_px > 0 else 0
                    else:
                        # Short: unrealized = (entry - mark) * |qty|
                        unrealized = (avg_px - yes_ask) * abs(pos) if avg_px > 0 else 0
                    total_pnl = realized + unrealized
                elif realized != 0:
                    total_pnl = realized
                else:
                    total_pnl = None

                if total_pnl is not None:
                    baseline = self._pnl_baseline.get(raw_strike, 0)
                    session_pnl = total_pnl - baseline
                    item.setText(f"${total_pnl:+.2f} (${session_pnl:+.2f})")
                    color = "#22c55e" if total_pnl > 0 else "#ef4444"
                    item.setForeground(QColor(color))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

    def _display_market_price(self, row: int, col: int,
                              price: float, size: int,
                              disp_strike: float, t_years: float,
                              base_color: str, highlight_price: float | None):
        """Display market price with implied vol sub-label (cols 2 or 3)."""
        existing = self.table.cellWidget(row, col)

        if price <= 0:
            if existing:
                self.table.removeCellWidget(row, col)
            item = self.table.item(row, col)
            if item:
                item.setText("--")
                item.setForeground(QColor("#5a6270"))
            return

        color = base_color
        if highlight_price is not None and abs(highlight_price - price) < 0.001:
            color = "#00e5ff"

        # Compute implied vol
        iv_text = ""
        if self.spot_price > 0 and t_years > 0:
            iv = _implied_vol_quadratic(price, self.spot_price, disp_strike, t_years,
                                       self.pricer.risk_free_rate)
            if iv > 0:
                iv_text = f"{iv * 100:.1f}%"

        if existing is None or existing.objectName() != "mkt_container":
            container = QWidget()
            container.setObjectName("mkt_container")
            container.setStyleSheet("background:transparent;")
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 1, 4, 1)
            lay.setSpacing(0)
            main_lbl = QLabel()
            main_lbl.setObjectName("mkt_main")
            iv_lbl = QLabel()
            iv_lbl.setObjectName("mkt_iv")
            iv_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            lay.addWidget(main_lbl)
            lay.addWidget(iv_lbl)
            self.table.setCellWidget(row, col, container)
            item = self.table.item(row, col)
            if item:
                item.setText("")
        else:
            container = existing

        main_lbl = container.findChild(QLabel, "mkt_main")
        iv_lbl = container.findChild(QLabel, "mkt_iv")

        main_lbl.setText(f"${price:.2f} ({size})")
        main_lbl.setStyleSheet(f"color:{color};font-size:12px;")

        iv_lbl.setText(iv_text)
        iv_lbl.setStyleSheet("color:#5a6270;font-size:8px;")

    def _display_greeks(self, row: int, raw_strike: float,
                        disp_strike: float, data: dict):
        """Update the Greek columns: col 9 (Δ), col 10 (γ), col 11 (ν), col 12 (θ).

        Delta = position * dP/dSpot  — $ change per $1 spot move.
        Gamma = position * d²P/dS²  — how fast delta changes per $1 move.
        Vega  = position * dP/dσ    — $ change per 1% IV move.
        Theta = position * dP/dT    — $ change per hour of time decay.
        """
        strat = self.strategies.get(raw_strike)
        pos = strat.position if strat else data.get("position", 0)

        dim = QColor("#5a6270")
        items = [self.table.item(row, c) for c in (9, 10, 11, 12)]

        if pos == 0 or not self.pricer.options or self.spot_price <= 0:
            for it in items:
                if it:
                    it.setText("--")
                    it.setForeground(dim)
            return

        close_time = ""
        if self.current_event:
            close_time = self.current_event.get("close_time", "")

        # Delta & gamma: bump spot ±$1
        bump = 1.0
        try:
            iv_p = self._get_iv_pricer()
            sigma_mid = iv_p._find_closest_bid_iv(disp_strike)
            p_mid = self.pricer.prob_above_with_iv(
                disp_strike, sigma_mid, spot=self.spot_price, kalshi_close_iso=close_time)
            p_up = self.pricer.prob_above_with_iv(
                disp_strike, sigma_mid, spot=self.spot_price + bump, kalshi_close_iso=close_time)
            p_dn = self.pricer.prob_above_with_iv(
                disp_strike, sigma_mid, spot=self.spot_price - bump, kalshi_close_iso=close_time)
            dp_ds = (p_up - p_dn) / (2 * bump)
            d2p_ds2 = (p_up - 2 * p_mid + p_dn) / (bump ** 2)
        except Exception:
            for it in items:
                if it:
                    it.setText("--")
                    it.setForeground(dim)
            return

        # Vega: bump IV ±1%
        vega_per = 0.0
        try:
            sigma = iv_p._find_closest_bid_iv(disp_strike)
            if sigma > 0:
                iv_bump = 0.01
                p_iv_up = self.pricer.prob_above_with_iv(
                    disp_strike, sigma + iv_bump,
                    spot=self.spot_price, kalshi_close_iso=close_time)
                p_iv_dn = self.pricer.prob_above_with_iv(
                    disp_strike, sigma - iv_bump,
                    spot=self.spot_price, kalshi_close_iso=close_time)
                vega_per = (p_iv_up - p_iv_dn) / (2 * iv_bump)
        except Exception:
            pass

        # Theta: price change per hour of time decay
        theta_per_hr = 0.0
        try:
            if sigma_mid > 0 and close_time:
                from datetime import datetime as _dt, timezone as _tz
                close_utc = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                now_ts = _dt.now(tz=_tz.utc).timestamp()
                T_sec = close_utc.timestamp() - now_ts
                if T_sec > 3600:  # more than 1 hour left
                    # Price 1 hour from now (shorter T)
                    one_hour_yr = 3600 / (365.25 * 24 * 3600)
                    T_now = T_sec / (365.25 * 24 * 3600)
                    T_later = T_now - one_hour_yr
                    p_later = self.pricer._bs_prob_above(
                        self.spot_price, disp_strike, sigma_mid, T_later, self.pricer.risk_free_rate)
                    theta_per_hr = p_later - p_mid  # negative for ATM (time decay)
        except Exception:
            pass

        pos_delta = pos * dp_ds
        pos_gamma = pos * d2p_ds2
        pos_vega = pos * vega_per * 0.01
        pos_theta = pos * theta_per_hr

        # Col 8: Delta
        if items[0]:
            items[0].setText(f"{pos_delta:+.4f}")
            items[0].setForeground(QColor("#22c55e" if pos_delta >= 0 else "#ef4444"))

        # Col 9: Gamma
        if items[1]:
            items[1].setText(f"{pos_gamma:+.5f}")
            items[1].setForeground(QColor("#f59e0b"))

        # Col 10: Vega
        if items[2]:
            items[2].setText(f"{pos_vega:+.5f}")
            items[2].setForeground(QColor("#8b5cf6"))

        # Col 11: Theta (per hour)
        if items[3]:
            items[3].setText(f"{pos_theta:+.4f}")
            items[3].setForeground(QColor("#06b6d4"))

    def _display_iv(self, row: int, disp_strike: float):
        """Update the Deribit IV cell (col 6) with bid/ask IV + strike used."""
        bid_iv, ask_iv, iv_strike = self._find_closest_iv(disp_strike)

        existing = self.table.cellWidget(row, 6)
        if existing is None:
            container = QWidget()
            container.setStyleSheet("background:transparent;")
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 1, 4, 1)
            lay.setSpacing(0)
            main_lbl = QLabel("--")
            main_lbl.setStyleSheet("color:#5a6270;font-size:12px;")
            main_lbl.setObjectName("iv_main")
            strike_lbl = QLabel("")
            strike_lbl.setStyleSheet("color:#3a4250;font-size:8px;")
            strike_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            strike_lbl.setObjectName("iv_strike")
            lay.addWidget(main_lbl)
            lay.addWidget(strike_lbl)
            self.table.setCellWidget(row, 6, container)
            item = self.table.item(row, 6)
            if item:
                item.setText("")
        else:
            container = existing

        main_lbl = container.findChild(QLabel, "iv_main")
        strike_lbl = container.findChild(QLabel, "iv_strike")

        if bid_iv is not None and ask_iv is not None:
            main_lbl.setText(f"{bid_iv:.1f}% / {ask_iv:.1f}%")
        elif ask_iv is not None:
            main_lbl.setText(f"-- / {ask_iv:.1f}%")
        else:
            main_lbl.setText("--")

        if iv_strike is not None:
            oi = self._get_iv_pricer()._find_closest_oi(disp_strike)
            if oi > 0:
                strike_lbl.setText(f"${iv_strike:,.0f} (OI:{oi:,.0f})")
            else:
                strike_lbl.setText(f"${iv_strike:,.0f}")
        else:
            strike_lbl.setText("")

    def _refit_vol_smile(self):
        """Fit a quadratic vol smile on mid IVs of strikes within 4% of spot.

        Updates self._smoothed_iv with EWM-smoothed fitted IVs (span=10).
        Called every 60s by timer, and on big spot moves from _on_price.
        """
        if self.spot_price <= 0 or not self.display_strikes:
            return

        # Compute T (time to expiry)
        t_years = 0.0
        if self.current_event:
            close_iso = self.current_event.get("close_time", "")
            if close_iso:
                try:
                    close_utc = datetime.fromisoformat(
                        close_iso.replace("Z", "+00:00"))
                    t_years = max(
                        (close_utc - datetime.now(tz=timezone.utc)).total_seconds()
                        / (365.25 * 86400), 0)
                except Exception:
                    pass
        if t_years <= 0:
            return

        # Compute mid IV for each strike from current market mids
        strikes_arr = []
        mid_ivs = []
        for raw_strike, disp in zip(self.strikes, self.display_strikes):
            data = self.market_data.get(raw_strike, {})
            bid = data.get("yes_bid", 0.0)
            ask = data.get("yes_ask", 0.0)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            iv = _implied_vol_quadratic(mid, self.spot_price, disp, t_years,
                                        self.pricer.risk_free_rate)
            if iv > 0:
                strikes_arr.append(disp)
                mid_ivs.append(iv)

        if len(strikes_arr) < 3:
            return

        strikes_np = np.array(strikes_arr)
        ivs_np = np.array(mid_ivs)

        # Filter to strikes within 4% of spot for fitting
        otm_pct = np.abs(strikes_np / self.spot_price - 1)
        nearby_mask = otm_pct < self._smile_otm_pct
        if nearby_mask.sum() < 3:
            return

        # Reject IV outliers using IQR on the nearby set
        nearby_ivs = ivs_np[nearby_mask]
        q1, q3 = np.percentile(nearby_ivs, [25, 75])
        iqr = q3 - q1
        iv_low = q1 - 1.5 * iqr
        iv_high = q3 + 1.5 * iqr
        inlier_mask = nearby_mask & (ivs_np >= iv_low) & (ivs_np <= iv_high)
        if inlier_mask.sum() < 3:
            return

        # Fit quadratic on nearby strikes (outliers removed)
        try:
            coeffs = np.polyfit(strikes_np[inlier_mask], ivs_np[inlier_mask], 2)
        except Exception:
            return
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        self._smile_coeffs = (a, b, c)

        # Evaluate fitted IV for ALL strikes and update EWM
        alpha = 2.0 / (self._smile_span + 1)
        for disp_k in strikes_arr:
            fitted = a * disp_k**2 + b * disp_k + c
            if fitted <= 0:
                continue
            prev = self._smoothed_iv.get(disp_k)
            if prev is not None and prev > 0:
                self._smoothed_iv[disp_k] = alpha * fitted + (1 - alpha) * prev
            else:
                self._smoothed_iv[disp_k] = fitted

        self._last_smile_fit_time = time.monotonic()
        self._last_smile_spot = self.spot_price

    def _display_smoothed_iv(self, row: int, disp_strike: float):
        """Update the Smoothed IV cell (col 7)."""
        item = self.table.item(row, 7)
        if not item:
            return
        siv = self._smoothed_iv.get(disp_strike)
        if siv and siv > 0:
            item.setText(f"{siv * 100:.1f}%")
            item.setForeground(QColor("#8b5cf6"))
        else:
            item.setText("--")
            item.setForeground(QColor("#5a6270"))

    _ui_dirty = False

    def _get_iv_pricer(self):
        """Return the pricer to use for IV lookup based on dropdown selection."""
        if self.iv_source_combo.currentText() == "Weekly" and self.weekly_pricer.options:
            return self.weekly_pricer
        return self.pricer

    def _recompute_and_trade(self):
        """Compute theos + feed strategies immediately. Runs on WS thread.

        Uses smoothed IV from vol smile fit when available, falls back to
        Deribit IV. Caches results for the UI timer to display.
        """
        if not self.strikes:
            return
        close_time = ""
        if self.current_event:
            close_time = self.current_event.get("close_time", "")
        spot_b = self.spot_bid if self.spot_bid > 0 else self.spot_price
        spot_a = self.spot_ask if self.spot_ask > 0 else self.spot_price
        if spot_b <= 0 or spot_a <= 0:
            return

        now = time.monotonic()

        # Fall back to Deribit IV if no smoothed IV available yet
        iv_pricer = self._get_iv_pricer()
        use_deribit_fallback = not self._smoothed_iv and iv_pricer.options

        for raw_strike, disp in zip(self.strikes, self.display_strikes):
            if use_deribit_fallback:
                # Deribit fallback — bid/ask IV → bid/ask theo
                bid_iv = iv_pricer._find_closest_bid_iv(disp)
                ask_iv = iv_pricer._find_closest_ask_iv(disp)
                bid_theo = self.pricer.prob_above_with_iv(
                    disp, bid_iv, spot=spot_b, kalshi_close_iso=close_time,
                )
                ask_theo = self.pricer.prob_above_with_iv(
                    disp, ask_iv, spot=spot_a, kalshi_close_iso=close_time,
                )
            else:
                # Smoothed IV — single IV → single theo (use spot bid/ask for conservatism)
                siv = self._smoothed_iv.get(disp, 0.0)
                if siv <= 0:
                    continue
                bid_theo = self.pricer.prob_above_with_iv(
                    disp, siv, spot=spot_b, kalshi_close_iso=close_time,
                )
                ask_theo = self.pricer.prob_above_with_iv(
                    disp, siv, spot=spot_a, kalshi_close_iso=close_time,
                )

            # Cache for UI
            self._cached_theos[raw_strike] = (bid_theo, ask_theo)

            # Track latency on actual computation
            last = self._theo_last_time.get(raw_strike)
            self._theo_last_time[raw_strike] = now
            if last is not None:
                dt_ms = (now - last) * 1000
                prev_ema = self._theo_latency_ema.get(raw_strike, dt_ms)
                self._theo_latency_ema[raw_strike] = 0.15 * dt_ms + 0.85 * prev_ema

            # Feed strategy immediately
            strat = self.strategies.get(raw_strike)
            if strat and strat.active:
                md = self.market_data.get(raw_strike)
                if md:
                    strat.kalshi_bid = md.get("yes_bid", 0.0)
                    strat.kalshi_ask = md.get("yes_ask", 0.0)
                    strat.kalshi_bid_size = md.get("bid_size", 0)
                    strat.kalshi_ask_size = md.get("ask_size", 0)
                    strat.avg_entry = md.get("avg_price", 0.0)
                strat.update_theo(bid_theo, ask_theo)

        # Feed stashed strategies (other events running in background)
        for et, stashed_strats in self._stashed_strategies.items():
            stashed_ev = self._stashed_events.get(et)
            stashed_md = self._stashed_market_data.get(et, {})
            if not stashed_ev:
                continue
            stash_close = stashed_ev.get("close_time", "")
            for raw_strike, strat in stashed_strats.items():
                if not strat.active:
                    continue
                disp = display_strike(raw_strike)
                if use_deribit_fallback:
                    stash_bid_iv = iv_pricer._find_closest_bid_iv(disp)
                    stash_ask_iv = iv_pricer._find_closest_ask_iv(disp)
                    bid_theo = self.pricer.prob_above_with_iv(
                        disp, stash_bid_iv, spot=spot_b, kalshi_close_iso=stash_close,
                    )
                    ask_theo = self.pricer.prob_above_with_iv(
                        disp, stash_ask_iv, spot=spot_a, kalshi_close_iso=stash_close,
                    )
                else:
                    siv = self._smoothed_iv.get(disp, 0.0)
                    if siv <= 0:
                        continue
                    bid_theo = self.pricer.prob_above_with_iv(
                        disp, siv, spot=spot_b, kalshi_close_iso=stash_close,
                    )
                    ask_theo = self.pricer.prob_above_with_iv(
                        disp, siv, spot=spot_a, kalshi_close_iso=stash_close,
                    )
                md = stashed_md.get(raw_strike)
                if md:
                    strat.kalshi_bid = md.get("yes_bid", 0.0)
                    strat.kalshi_ask = md.get("yes_ask", 0.0)
                    strat.kalshi_bid_size = md.get("bid_size", 0)
                    strat.kalshi_ask_size = md.get("ask_size", 0)
                    strat.avg_entry = md.get("avg_price", 0.0)
                strat.update_theo(bid_theo, ask_theo)

    def _update_fast_theos(self):
        """UI timer (200ms) — refresh theo display from cached values."""
        if not self._ui_dirty:
            return
        self._ui_dirty = False

        for row, raw_strike in enumerate(self.strikes):
            if row >= self.table.rowCount():
                break
            cached = self._cached_theos.get(raw_strike)
            if cached:
                bid_theo, ask_theo = cached
                latency_ms = self._theo_latency_ema.get(raw_strike)
                self._display_theo_cached(row, bid_theo, ask_theo, latency_ms)

    def _display_theo_cached(self, row: int, bid_theo: float, ask_theo: float,
                             latency_ms: float | None):
        """Update the Theo cell from cached values."""
        existing = self.table.cellWidget(row, 4)
        if existing is None:
            container = QWidget()
            container.setStyleSheet("background:transparent;")
            lay = QVBoxLayout(container)
            lay.setContentsMargins(4, 1, 4, 1)
            lay.setSpacing(0)
            main_lbl = QLabel("--")
            main_lbl.setStyleSheet("color:#c8cdd5;font-size:12px;")
            main_lbl.setObjectName("theo_main")
            lat_lbl = QLabel("")
            lat_lbl.setStyleSheet("color:#3a4250;font-size:8px;")
            lat_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            lat_lbl.setObjectName("theo_lat")
            lay.addWidget(main_lbl)
            lay.addWidget(lat_lbl)
            self.table.setCellWidget(row, 4, container)
            item = self.table.item(row, 4)
            if item:
                item.setText("")
        else:
            container = existing

        main_lbl = container.findChild(QLabel, "theo_main")
        lat_lbl = container.findChild(QLabel, "theo_lat")

        main_lbl.setText(f"${bid_theo:.3f} / ${ask_theo:.3f}")
        main_lbl.setStyleSheet("color:#c8cdd5;font-size:12px;")

        if latency_ms is not None:
            lat_lbl.setText(f"{latency_ms:.0f}ms")
        else:
            lat_lbl.setText("")

    def _display_orders(self, row: int, strat, raw_strike: float = 0):
        """Update Order cell (col 13). Flatten orders are highlighted orange."""
        # Use market_data position as source of truth (updated by WS fills + REST)
        data = self.market_data.get(raw_strike, {})
        pos = data.get("position", 0)
        if strat:
            pos = strat.position
        buy_flatten = pos < 0
        sell_flatten = pos > 0

        has_buy = strat and strat.current_buy_price is not None
        has_sell = strat and strat.current_sell_price is not None

        if not has_buy and not has_sell:
            # No orders — use plain item
            existing = self.table.cellWidget(row, 13)
            if existing:
                self.table.removeCellWidget(row, 13)
            item = self.table.item(row, 13)
            if item:
                item.setText("--")
                item.setForeground(QColor("#5a6270"))
            return

        # Build widget with colored labels per side
        existing = self.table.cellWidget(row, 13)
        if existing is None or existing.objectName() != "order_container":
            container = QWidget()
            container.setObjectName("order_container")
            container.setStyleSheet("background:transparent;")
            hlay = QHBoxLayout(container)
            hlay.setContentsMargins(4, 1, 4, 1)
            hlay.setSpacing(4)
            buy_lbl = QLabel("")
            buy_lbl.setObjectName("order_buy")
            sell_lbl = QLabel("")
            sell_lbl.setObjectName("order_sell")
            hlay.addWidget(buy_lbl)
            hlay.addWidget(sell_lbl)
            self.table.setCellWidget(row, 13, container)
            item = self.table.item(row, 13)
            if item:
                item.setText("")
        else:
            container = existing

        buy_lbl = container.findChild(QLabel, "order_buy")
        sell_lbl = container.findChild(QLabel, "order_sell")

        # Normal = yellow (#facc15), Flatten = bright orange (#ff8c00) + bold
        normal = "color:#facc15;font-size:12px;"
        flatten = "color:#ff8c00;font-size:12px;font-weight:bold;"

        if has_buy:
            buy_size = abs(pos) if buy_flatten else strat.size_bid
            buy_lbl.setText(f"B${strat.current_buy_price:.2f}({buy_size})")
            buy_lbl.setStyleSheet(flatten if buy_flatten else normal)
        else:
            buy_lbl.setText("")

        if has_sell:
            sell_size = pos if sell_flatten else strat.size_ask
            sell_lbl.setText(f"S${strat.current_sell_price:.2f}({sell_size})")
            sell_lbl.setStyleSheet(flatten if sell_flatten else normal)
        else:
            sell_lbl.setText("")

    # =========================================================================
    # Strategy Controls
    # =========================================================================

    def _on_strike_clicked(self, row: int, col: int):
        """Double-click a row to open per-strike params dialog."""
        if row < 0 or row >= len(self.strikes):
            return
        raw_strike = self.strikes[row]
        disp = self.display_strikes[row]

        # Get current params from strategy if exists, else saved, else defaults
        strat = self.strategies.get(raw_strike)
        if strat:
            eb, ea = strat.edge_bid, strat.edge_ask
            sb, sa = strat.size_bid, strat.size_ask
            max_pos, tol = strat.max_position, strat.tolerance
            fwi, fws = strat.flatten_walk_interval, strat.flatten_walk_step
        else:
            saved = _load_strategy_params().get(str(raw_strike), {})
            eb = saved.get("edge_bid", 0.03)
            ea = saved.get("edge_ask", 0.03)
            sb = saved.get("size_bid", 10)
            sa = saved.get("size_ask", 10)
            max_pos = saved.get("max_position", 50)
            tol = saved.get("tolerance", 0.01)
            fwi = saved.get("flatten_walk_interval", 0.0)
            fws = saved.get("flatten_walk_step", 0.01)

        ae_on = self.auto_edge_btn.isChecked()
        dlg = StrikeParamsDialog(f"${disp:,.0f}", eb, ea, sb, sa, max_pos, tol, fwi, fws, self, auto_edge=ae_on)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            params = dlg.get_params()
            if params is None:
                return
            new_eb, new_ea, new_sb, new_sa, new_max, new_tol, new_fwi, new_fws = params

            if strat is None:
                data = self.market_data.get(raw_strike)
                if not data:
                    return
                # When auto edge is on, use 0 as placeholder — auto edge will set real values
                init_eb = new_eb if not ae_on else 0.0
                init_ea = new_ea if not ae_on else 0.0
                strat = Strategy(
                    ticker=data["ticker"], strike=disp,
                    edge_bid=init_eb, edge_ask=init_ea,
                    size_bid=new_sb, size_ask=new_sa,
                    max_position=new_max, api=self.api,
                    tolerance=new_tol, on_max_position=self._on_max_position,
                )
                strat.flatten_walk_interval = new_fwi
                strat.flatten_walk_step = new_fws
                self.strategies[raw_strike] = strat
                if ae_on:
                    self._recompute_auto_edges(force=True)
            else:
                if ae_on:
                    # Auto edge controls edge — only update non-edge params
                    strat.size_bid = new_sb
                    strat.size_ask = new_sa
                    strat.max_position = new_max
                    strat.tolerance = new_tol
                    strat.flatten_walk_interval = new_fwi
                    strat.flatten_walk_step = new_fws
                else:
                    old_eb, old_ea = strat.edge_bid, strat.edge_ask
                    strat.update_params(new_eb, new_ea, new_sb, new_sa, new_max, new_tol,
                                        new_fwi, new_fws)
                    # If edge changed and strategy is active, force reprice
                    if strat.active and (new_eb != old_eb or new_ea != old_ea):
                        strat._cancel_sell()
                        strat._cancel_buy()
                        strat.current_sell_price = None
                        strat.current_buy_price = None

            self._save_all_strategy_params()
            print(f"[App] ${disp:,.0f} params: edge_bid={new_eb}, edge_ask={new_ea}, "
                  f"size_bid={new_sb}, size_ask={new_sa}, max_pos={new_max}, tol={new_tol}, "
                  f"walk_int={new_fwi}s, walk_step=${new_fws}")

    def _on_table_right_click(self, pos):
        """Right-click context menu for selected rows."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#141923;color:#c8cdd5;border:1px solid #1e2736;}"
            "QMenu::item:selected{background:#1e2736;}"
        )
        on_action = menu.addAction(f"Turn ON ({len(rows)} selected)")
        off_action = menu.addAction(f"Turn OFF ({len(rows)} selected)")
        params_action = menu.addAction(f"Edit Params ({len(rows)} selected)")

        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == on_action:
            self._toggle_selected(True)
        elif action == off_action:
            self._toggle_selected(False)
        elif action == params_action:
            self._edit_selected_params(rows)

    def _toggle_selected(self, start: bool):
        """Start or stop strategies for all selected rows."""
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        for row in rows:
            if row < 0 or row >= len(self.strikes):
                continue
            raw_strike = self.strikes[row]
            strat = self.strategies.get(raw_strike)
            if start and (strat is None or not strat.active):
                self._toggle_strategy(raw_strike)
            elif not start and strat and strat.active:
                self._toggle_strategy(raw_strike)

    def _edit_selected_params(self, rows: list[int]):
        """Open a bulk params dialog for multiple selected strikes."""
        first_raw = self.strikes[rows[0]]
        strat = self.strategies.get(first_raw)
        if strat:
            eb, ea = strat.edge_bid, strat.edge_ask
            sb, sa = strat.size_bid, strat.size_ask
            max_pos, tol = strat.max_position, strat.tolerance
            fwi, fws = strat.flatten_walk_interval, strat.flatten_walk_step
        else:
            saved = _load_strategy_params().get(str(first_raw), {})
            eb = saved.get("edge_bid", 0.03)
            ea = saved.get("edge_ask", 0.03)
            sb = saved.get("size_bid", 10)
            sa = saved.get("size_ask", 10)
            max_pos = saved.get("max_position", 50)
            tol = saved.get("tolerance", 0.01)
            fwi = saved.get("flatten_walk_interval", 0.0)
            fws = saved.get("flatten_walk_step", 0.01)

        ae_on = self.auto_edge_btn.isChecked()
        dlg = StrikeParamsDialog(
            f"{len(rows)} strikes", eb, ea, sb, sa, max_pos, tol, fwi, fws, self, auto_edge=ae_on
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = dlg.get_params()
        if params is None:
            return
        new_eb, new_ea, new_sb, new_sa, new_max, new_tol, new_fwi, new_fws = params

        for row in rows:
            if row < 0 or row >= len(self.strikes):
                continue
            raw_strike = self.strikes[row]
            data = self.market_data.get(raw_strike)
            if not data:
                continue
            disp = self.display_strikes[row]

            strat = self.strategies.get(raw_strike)
            if strat is None:
                init_eb = new_eb if not ae_on else 0.0
                init_ea = new_ea if not ae_on else 0.0
                strat = Strategy(
                    ticker=data["ticker"], strike=disp,
                    edge_bid=init_eb, edge_ask=init_ea,
                    size_bid=new_sb, size_ask=new_sa,
                    max_position=new_max, api=self.api,
                    tolerance=new_tol, on_max_position=self._on_max_position,
                )
                strat.flatten_walk_interval = new_fwi
                strat.flatten_walk_step = new_fws
                self.strategies[raw_strike] = strat
            else:
                if ae_on:
                    # Auto edge controls edge — only update non-edge params
                    strat.size_bid = new_sb
                    strat.size_ask = new_sa
                    strat.max_position = new_max
                    strat.tolerance = new_tol
                    strat.flatten_walk_interval = new_fwi
                    strat.flatten_walk_step = new_fws
                else:
                    old_eb, old_ea = strat.edge_bid, strat.edge_ask
                    strat.update_params(new_eb, new_ea, new_sb, new_sa, new_max, new_tol,
                                        new_fwi, new_fws)
                    if strat.active and (new_eb != old_eb or new_ea != old_ea):
                        strat._cancel_sell()
                        strat._cancel_buy()
                        strat.current_sell_price = None
                        strat.current_buy_price = None

        if ae_on:
            self._recompute_auto_edges(force=True)

        self._save_all_strategy_params()
        print(f"[App] Bulk params ({len(rows)} strikes): edge_bid={new_eb}, "
              f"edge_ask={new_ea}, size_bid={new_sb}, size_ask={new_sa}, "
              f"max_pos={new_max}, tol={new_tol}, walk_int={new_fwi}s, walk_step=${new_fws}")

    def _save_all_strategy_params(self):
        """Persist all strategy params to disk, keyed by strike price.

        This allows settings to survive event transitions — the same
        strike in a new event inherits the previous settings.
        """
        params = {}
        for raw_strike, strat in self.strategies.items():
            strike_key = str(raw_strike)
            params[strike_key] = {
                "edge_bid": strat.edge_bid,
                "edge_ask": strat.edge_ask,
                "size_bid": strat.size_bid,
                "size_ask": strat.size_ask,
                "max_position": strat.max_position,
                "tolerance": strat.tolerance,
                "flatten_walk_interval": strat.flatten_walk_interval,
                "flatten_walk_step": strat.flatten_walk_step,
            }
        _save_strategy_params(params)

    def _on_max_position(self, ticker: str):
        """Called by Strategy when max position is hit. Show MAX on button."""
        for raw_strike, data in self.market_data.items():
            if data.get("ticker") == ticker:
                row = self.strikes.index(raw_strike) if raw_strike in self.strikes else -1
                if row >= 0:
                    btn = self.table.cellWidget(row, 15)
                    if btn:
                        btn.setText("MAX")
                        btn.setStyleSheet(
                            "QPushButton{background:#1e2736;color:#ff8c00;border:1px solid #ff8c00;"
                            "border-radius:3px;padding:2px 6px;font-size:11px;}"
                            "QPushButton:hover{background:#2d3a4d;}"
                        )
                disp = display_strike(raw_strike)
                print(f"[App] ${disp:,.0f} MAX POSITION — widening edge by $0.10")
                break

    def _toggle_strategy(self, raw_strike: float):
        """Turn strategy ON/OFF for one strike."""
        data = self.market_data.get(raw_strike)
        if not data:
            return
        ticker = data["ticker"]
        disp = display_strike(raw_strike)

        strat = self.strategies.get(raw_strike)
        if strat and strat.active:
            # Turn OFF
            strat.stop()
            print(f"[App] Strategy OFF for ${disp:,.0f}")
        else:
            # Block re-enable if already at max position
            if strat and abs(strat.position) >= strat.max_position:
                QMessageBox.warning(
                    self, "Max Position",
                    f"${disp:,.0f} is at max position ({strat.position}). "
                    f"Cannot re-enable strategy."
                )
                return

            # Turn ON — create if needed, using saved params or defaults
            if strat is None:
                saved = _load_strategy_params().get(str(raw_strike), {})
                strat = Strategy(
                    ticker=ticker, strike=disp,
                    edge_bid=saved.get("edge_bid", 0.03),
                    edge_ask=saved.get("edge_ask", 0.03),
                    size_bid=saved.get("size_bid", 10),
                    size_ask=saved.get("size_ask", 10),
                    max_position=saved.get("max_position", 50),
                    api=self.api,
                    tolerance=saved.get("tolerance", 0.01),
                    on_max_position=self._on_max_position,
                )
                strat.flatten_walk_interval = saved.get("flatten_walk_interval", 0.0)
                strat.flatten_walk_step = saved.get("flatten_walk_step", 0.01)
                self.strategies[raw_strike] = strat

            # Validate sizes <= max_position
            if strat.size_bid > strat.max_position or strat.size_ask > strat.max_position:
                QMessageBox.warning(
                    self, "Invalid Size",
                    f"Size (bid={strat.size_bid}, ask={strat.size_ask}) exceeds "
                    f"max position ({strat.max_position}). "
                    f"Please adjust parameters before enabling."
                )
                return

            strat.start()
            print(f"[App] Strategy ON for ${disp:,.0f}")

        # Update button appearance
        row = self.strikes.index(raw_strike) if raw_strike in self.strikes else -1
        if row >= 0:
            btn = self.table.cellWidget(row, 15)
            if btn:
                if strat.active:
                    btn.setText("ON")
                    btn.setStyleSheet(
                        "QPushButton{background:#1e2736;color:#22c55e;border:1px solid #22c55e;"
                        "border-radius:3px;padding:2px 6px;font-size:11px;}"
                        "QPushButton:hover{background:#2d3a4d;}"
                    )
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet(
                        "QPushButton{background:#1e2736;color:#ef4444;border:1px solid #2d3a4d;"
                        "border-radius:3px;padding:2px 6px;font-size:11px;}"
                        "QPushButton:hover{background:#2d3a4d;}"
                    )

    def _flatten_position(self, raw_strike: float):
        """Immediately flatten position by taking the other side BBO."""
        strat = self.strategies.get(raw_strike)
        md = self.market_data.get(raw_strike, {})
        ticker = md.get("ticker", "")
        pos = 0

        if strat:
            pos = strat.position
        else:
            # No strategy — check market_data for position
            pos = md.get("position", 0)

        if pos == 0 or not ticker:
            print(f"[App] Flatten: no position to flatten for {display_strike(raw_strike):,.0f}")
            return

        # Stop the strategy to cancel resting orders first
        if strat and strat.active:
            strat.stop()

        try:
            if pos > 0:
                # Long → sell at market bid (hit the bid)
                price = strat.kalshi_bid if strat else md.get("yes_bid", 0)
                if price <= 0:
                    print(f"[App] Flatten: no bid available")
                    return
                resp = self.api.create_order(
                    ticker=ticker, side="yes", action="sell",
                    price_dollars=f"{price:.2f}", count=abs(pos),
                )
                print(f"[App] Flatten: SELL {abs(pos)}x @ ${price:.2f} "
                      f"(${display_strike(raw_strike):,.0f})")
            else:
                # Short → buy at market ask (lift the ask)
                price = strat.kalshi_ask if strat else md.get("yes_ask", 0)
                if price <= 0:
                    print(f"[App] Flatten: no ask available")
                    return
                resp = self.api.create_order(
                    ticker=ticker, side="yes", action="buy",
                    price_dollars=f"{price:.2f}", count=abs(pos),
                )
                print(f"[App] Flatten: BUY {abs(pos)}x @ ${price:.2f} "
                      f"(${display_strike(raw_strike):,.0f})")
        except Exception as e:
            print(f"[App] Flatten failed: {e}")

    def _restore_strategies(self):
        """Rebuild strategy instances from resting orders on the exchange."""
        try:
            orders = self.api.get_orders(status="resting")
        except Exception as e:
            print(f"[App] Restore orders failed: {e}")
            return

        tickers = {d["ticker"]: raw for raw, d in self.market_data.items()}
        saved_params = _load_strategy_params()

        for o in orders:
            ticker = o.get("ticker", "")
            if ticker not in tickers:
                continue
            raw_strike = tickers[ticker]
            disp = display_strike(raw_strike)
            params = saved_params.get(str(raw_strike), {})

            # Create strategy if not already restored
            strat = self.strategies.get(raw_strike)
            if strat is None:
                strat = Strategy(
                    ticker=ticker, strike=disp,
                    edge_bid=params.get("edge_bid", 0.03),
                    edge_ask=params.get("edge_ask", 0.03),
                    size_bid=params.get("size_bid", 10),
                    size_ask=params.get("size_ask", 10),
                    max_position=params.get("max_position", 50),
                    api=self.api,
                    tolerance=params.get("tolerance", 0.01),
                    on_max_position=self._on_max_position,
                )
                strat.flatten_walk_interval = params.get("flatten_walk_interval", 0.0)
                strat.flatten_walk_step = params.get("flatten_walk_step", 0.01)
                strat.start()
                self.strategies[raw_strike] = strat

            # Restore order to the correct side
            action = o.get("action", "")
            order_id = o.get("order_id")
            price_str = o.get("yes_price_dollars") or o.get("yes_price", "0")
            price = float(price_str) if price_str else None
            if action == "sell":
                strat.resting_sell_id = order_id
                strat.current_sell_price = price
                strat.ask_active = True
            else:
                strat.resting_buy_id = order_id
                strat.current_buy_price = price
                strat.bid_active = True

            # Update toggle button
            row = self.strikes.index(raw_strike) if raw_strike in self.strikes else -1
            if row >= 0:
                btn = self.table.cellWidget(row, 15)
                if btn:
                    btn.setText("ON")
                    btn.setStyleSheet(
                        "QPushButton{background:#1e2736;color:#22c55e;"
                        "border:1px solid #22c55e;border-radius:3px;"
                        "padding:2px 6px;font-size:11px;}"
                        "QPushButton:hover{background:#2d3a4d;}"
                    )
            print(f"[App] Restored ${disp:,.0f} {action} order={order_id} @ ${price}")

    def _refresh_positions(self):
        """Fetch positions + fills from Kalshi and update market_data + strategies."""
        try:
            positions = self.api.get_positions()
        except Exception as e:
            print(f"[App] Position fetch failed: {e}")
            return

        # Build ticker -> position data map
        pos_map = {p.get("ticker", ""): p for p in positions}

        # Fetch fills and compute realized PnL + avg fill price per ticker
        pnl_map = {}
        avg_price_map = {}  # ticker -> weighted avg yes_price from fills
        try:
            tickers = {d["ticker"] for d in self.market_data.values()}
            fills = self.api.get_fills()
            if fills and not hasattr(self, '_fills_logged'):
                print(f"[App] Sample fill keys: {list(fills[0].keys())}")
                print(f"[App] Sample fill: {fills[0]}")
                self._fills_logged = True

            # Walk fills chronologically to compute:
            #   realized PnL  — profit from closed round trips only
            #   cost_basis     — total cost of the current open position (always >= 0)
            #   avg fill price — cost_basis / abs(position)
            #
            # Model: track running position + avg entry price.
            # When reducing, realize PnL = (exit - entry) * closed_qty (long)
            #                  or PnL = (entry - exit) * closed_qty (short).
            # When crossing zero, realize the full old side, open new side at fill price.
            run_pos = {}       # ticker -> running position (signed int)
            avg_entry = {}     # ticker -> avg entry price of current position
            realized_pnl = {}  # ticker -> realized PnL from closed trades

            fills_sorted = sorted(fills, key=lambda f: f.get("created_time", ""))

            for f in fills_sorted:
                t = f.get("ticker", "")
                if t not in tickers:
                    continue
                action = f.get("action", "")
                count = int(float(f.get("count_fp", 0)))
                yes_price = float(f.get("yes_price_dollars", 0) or 0)

                if t not in run_pos:
                    run_pos[t] = 0
                    avg_entry[t] = 0.0
                    realized_pnl[t] = 0.0

                prev = run_pos[t]
                delta = count if action == "buy" else -count
                new = prev + delta

                if prev == 0:
                    # Opening from flat
                    avg_entry[t] = yes_price

                elif (prev > 0 and delta > 0) or (prev < 0 and delta < 0):
                    # Adding to same direction — update weighted avg entry
                    total_cost = avg_entry[t] * abs(prev) + yes_price * abs(delta)
                    avg_entry[t] = total_cost / abs(new)

                else:
                    # Reducing or flipping
                    closed = min(abs(delta), abs(prev))
                    if prev > 0:
                        # Long closed by selling: PnL = (sell - avg_buy) * qty
                        realized_pnl[t] += (yes_price - avg_entry[t]) * closed
                    else:
                        # Short closed by buying: PnL = (avg_sell - buy) * qty
                        realized_pnl[t] += (avg_entry[t] - yes_price) * closed

                    if new == 0:
                        avg_entry[t] = 0.0
                    elif (prev > 0 and new < 0) or (prev < 0 and new > 0):
                        # Crossed zero — new position opens at this fill price
                        avg_entry[t] = yes_price
                    # else: reduced but same direction, avg_entry stays the same

                run_pos[t] = new

            # Build output maps
            for t in run_pos:
                pnl_map[t] = realized_pnl.get(t, 0.0)
                if run_pos[t] != 0 and avg_entry.get(t, 0) > 0:
                    avg_price_map[t] = avg_entry[t]
        except Exception as e:
            print(f"[App] Fills fetch failed: {e}")

        # Update market_data (always) and strategy (if exists)
        for raw_strike, data in self.market_data.items():
            ticker = data["ticker"]
            p = pos_map.get(ticker, {})
            pos = int(float(p.get("position_fp", 0)))
            exposure = float(p.get("market_exposure_dollars", 0))
            pnl = pnl_map.get(ticker, 0.0)
            avg_px = avg_price_map.get(ticker, 0.0)
            data["position"] = pos
            data["exposure"] = exposure
            data["realized_pnl"] = pnl
            data["avg_price"] = avg_px

            # Capture baseline PnL on first refresh for session tracking
            if raw_strike not in self._pnl_baseline and pnl != 0:
                self._pnl_baseline[raw_strike] = pnl

            strat = self.strategies.get(raw_strike)
            if strat:
                strat.position = pos
                strat.exposure = exposure
                strat.realized_pnl = pnl

    def _my_tickers(self) -> set:
        """Return all tickers this instance manages (current + stashed events)."""
        tickers = {d["ticker"] for d in self.market_data.values()}
        for stashed_md in self._stashed_market_data.values():
            tickers.update(d["ticker"] for d in stashed_md.values())
        return tickers

    def _audit_orders(self):
        """Check resting orders, cancel orphans (>2 per ticker). Only touches this instance's tickers."""
        try:
            orders = self.api.get_orders(status="resting")
            if not orders:
                return

            my_tickers = self._my_tickers()

            # Group by ticker (only ours)
            by_ticker = {}
            for o in orders:
                t = o.get("ticker", "unknown")
                if t not in my_tickers:
                    continue
                if t not in by_ticker:
                    by_ticker[t] = []
                by_ticker[t].append(o)

            total = sum(len(ol) for ol in by_ticker.values())
            parts = [f"{t}: {len(ol)}" for t, ol in sorted(by_ticker.items())]
            print(f"[Audit] {total} resting orders (ours) — {', '.join(parts)}")

            # Build set of known strategy order IDs
            known_ids = set()
            for strat in self.strategies.values():
                if strat.resting_sell_id:
                    known_ids.add(strat.resting_sell_id)
                if strat.resting_buy_id:
                    known_ids.add(strat.resting_buy_id)
            for stashed in self._stashed_strategies.values():
                for strat in stashed.values():
                    if strat.resting_sell_id:
                        known_ids.add(strat.resting_sell_id)
                    if strat.resting_buy_id:
                        known_ids.add(strat.resting_buy_id)

            # Cancel any order on our tickers that we don't recognize
            orphans = 0
            for ticker_orders in by_ticker.values():
                for o in ticker_orders:
                    oid = o.get("order_id", "")
                    if oid and oid not in known_ids:
                        try:
                            self.api.cancel_order(oid)
                            orphans += 1
                            print(f"[Audit] Cancelled orphan {oid} ({o.get('ticker', '')} "
                                  f"{o.get('action', '')} @ {o.get('yes_price_dollars', '')})")
                        except Exception as e:
                            print(f"[Audit] Failed to cancel orphan {oid}: {e}")

            if orphans:
                print(f"[Audit] Cleaned up {orphans} orphan orders")
        except Exception as e:
            print(f"[Audit] Order check failed: {e}")

    def _stop_all_strategies(self):
        """Stop all active strategies and clear the dict."""
        for strat in self.strategies.values():
            if strat.active:
                strat.stop()
        self.strategies.clear()

    def _cancel_all_orders(self):
        """Cancel all resting orders belonging to this instance's tickers only."""
        try:
            orders = self.api.get_orders(status="resting")
            if not orders:
                print("[App] No resting orders to cancel")
                return

            my_tickers = self._my_tickers()
            my_orders = [o for o in orders if o.get("ticker", "") in my_tickers]

            if not my_orders:
                print("[App] No resting orders for our tickers")
                return

            cancelled = 0
            failed = 0
            for o in my_orders:
                oid = o.get("order_id", "")
                try:
                    self.api.cancel_order(oid)
                    cancelled += 1
                except Exception as e:
                    failed += 1
                    print(f"[App] Cancel failed {oid}: {e}")
            skipped = len(orders) - len(my_orders)
            print(f"[App] Cancelled {cancelled}/{len(my_orders)} orders "
                  f"({failed} failed, {skipped} skipped from other instances)")

            # Retry any that failed
            if failed > 0:
                time.sleep(0.5)
                remaining = self.api.get_orders(status="resting")
                for o in remaining:
                    if o.get("ticker", "") not in my_tickers:
                        continue
                    oid = o.get("order_id", "")
                    try:
                        self.api.cancel_order(oid)
                        print(f"[App] Retry cancelled {oid}")
                    except Exception:
                        pass
                final = [o for o in self.api.get_orders(status="resting")
                         if o.get("ticker", "") in my_tickers]
                if final:
                    print(f"[App] WARNING: {len(final)} of our orders still resting after retry!")
                else:
                    print("[App] All our orders cancelled on retry")
        except Exception as e:
            print(f"[App] Failed to fetch resting orders: {e}")

    # =========================================================================
    # Stylesheet
    # =========================================================================

    def _apply_stylesheet(self):
        self.setStyleSheet("""
        QMainWindow{background:#0b0f19;}
        QWidget{background:#0b0f19;color:#c8cdd5;font-size:12px;}
        QLabel{color:#c8cdd5;}
        QComboBox{background:#141923;color:#c8cdd5;border:1px solid #1e2736;
                  border-radius:3px;padding:4px 8px;}
        QComboBox::drop-down{border:none;}
        QComboBox QAbstractItemView{background:#141923;color:#c8cdd5;
                                    selection-background-color:#1e2736;}
        QTableWidget{background:#0b0f19;gridline-color:#1e2736;
                     border:1px solid #1e2736;color:#c8cdd5;font-size:12px;}
        QHeaderView::section{background:#141923;color:#5a6270;
                             border:1px solid #1e2736;padding:4px;font-weight:bold;}
        QTableWidget::item{padding:3px 6px;}
        QTableWidget::item:selected{background:#1e2736;}
        """)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def closeEvent(self, event):
        # 1. Stop feeds FIRST so no new orders can be placed
        if self.price_feed:
            self.price_feed.stop()
        if self.ws_feed:
            self.ws_feed.stop()
        if self.deribit_ws:
            self.deribit_ws.stop()
        if self.weekly_deribit_ws:
            self.weekly_deribit_ws.stop()

        # 2. Stop all strategies (cancels their tracked orders)
        all_strats = list(self.strategies.values())
        for stashed in self._stashed_strategies.values():
            all_strats.extend(stashed.values())
        for strat in all_strats:
            strat.active = False
            strat.bid_active = False
            strat.ask_active = False
            strat._cancel_sell()
            strat._cancel_buy()

        # 3. Sweep ALL resting orders as a safety net — catches anything
        #    the strategies didn't track (e.g. orders from a previous session)
        self._cancel_all_orders()

        self.strategies.clear()
        self._stashed_strategies.clear()
        self._stashed_market_data.clear()
        self._stashed_events.clear()
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window = AboveBelowApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
