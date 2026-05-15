"""
Position manager — PyQt app for viewing trades across both product
families:

  • Daily 5pm strike-ladder events (KXBTCD-* etc.)
  • 15-minute up/down markets (KXBTC15M-* etc., written by Aston)

The schema is identical (legacy `fills` table), so the trade view +
P&L breakdown works for both.  A top-level mode toggle controls which
file family the event picker shows.  In 15-min mode the grouping is by
day per series, and a ticker picker lets you drill into a specific
window or view the whole day at once.

Reads from analysis/backtesting/data/*.db.

Usage:
    python position_manager.py
"""

import sys
import sqlite3
from pathlib import Path

import pandas as pd

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "analysis" / "backtesting" / "data"
ICON_PATH = Path(__file__).resolve().parent / "icon.png"
TZ_LOCAL = "America/Chicago"


def event_db_path(event_ticker: str) -> Path:
    return DATA_DIR / f"{event_ticker}.db"


def to_ct(series):
    """Convert UTC timestamp series to Central Time (naive)."""
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)


# ---------------------------------------------------------------------------
# File discovery — split by product family
# ---------------------------------------------------------------------------

# 15-min files Aston writes are named `<SERIES>15M-<DAY>.db`, e.g.
# `KXBTC15M-26MAY.db`.  Anything else (KXBTCD, KXETHD, …) we treat as
# daily 5pm strike-ladder.

def _list_all_db_stems() -> list[str]:
    rows = []
    for db_file in DATA_DIR.glob("*.db"):
        if db_file.name == "recorder.db":
            continue
        rows.append(db_file.stem)
    return sorted(rows)


def list_daily_events() -> list[str]:
    """Stems for daily 5pm events — anything without `15M` in the name."""
    return [s for s in _list_all_db_stems() if "15M" not in s]


def list_15m_files() -> list[str]:
    """Stems for 15-min files — anything with `15M` in the name."""
    return [s for s in _list_all_db_stems() if "15M" in s]


def split_15m_stem(stem: str) -> tuple[str, str]:
    """`KXBTC15M-26MAY14` → ('KXBTC15M', '26MAY14').  Empty strings on
    malformed input — caller falls back gracefully."""
    if "-" not in stem:
        return stem, ""
    series, _, day = stem.rpartition("-")
    return series, day


_MONTH_CODES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def parse_day_label(label: str):
    """Decode a `YYMONDD` suffix (e.g. `26MAY14`) into a real date.

    Used to sort day options chronologically (lexicographic on the
    raw label breaks across months: `APR` < `AUG` alphabetically but
    August comes later in the calendar) and to display them in a
    human-readable `YYYY-MM-DD` form.  Returns None on parse failure
    so legacy / malformed names sink to the bottom rather than crash."""
    if not label or len(label) < 7:
        return None
    try:
        yy = int(label[:2])
        mon = label[2:5]
        dd = int(label[5:])
        if mon not in _MONTH_CODES:
            return None
        month = _MONTH_CODES.index(mon) + 1
        from datetime import date
        return date(2000 + yy, month, dd)
    except (ValueError, IndexError):
        return None


def display_day_label(label: str) -> str:
    """Render `26MAY14` as `2026-05-14`; pass through unrecognized
    labels unchanged so legacy names still appear in the dropdown."""
    d = parse_day_label(label)
    return d.isoformat() if d else label


# ---------------------------------------------------------------------------
# Fill loading + per-row P&L computation
# ---------------------------------------------------------------------------

def load_fills(stem: str) -> pd.DataFrame:
    if not stem:
        return pd.DataFrame()
    db_file = event_db_path(stem)
    if not db_file.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_file))
    try:
        df = pd.read_sql("SELECT * FROM fills ORDER BY ts", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def compute_trade_view(fills: pd.DataFrame) -> pd.DataFrame:
    """Walk fills chronologically; compute position + cumulative P&L per row.

    Position and avg-entry are tracked PER-TICKER so a 15-min window's
    state doesn't leak into the next window's — each ticker is its own
    contract.  Cumulative realized P&L and fees stay session-wide
    (the natural "total made today" summary)."""
    if fills.empty:
        return pd.DataFrame()

    fills = fills.sort_values("ts").reset_index(drop=True)
    rows = []
    # Per-ticker position state.  Each entry is (pos, avg_entry).
    ticker_state: dict[str, tuple[float, float]] = {}
    cum_realized = 0.0
    cum_fees = 0.0

    for _, f in fills.iterrows():
        action = f["action"]
        size = float(f["count"])
        price = float(f["price"])
        fee = float(f.get("fee") or 0.0)
        ts = f["ts"]
        ticker_key = f.get("ticker", "") or "(unknown)"

        prev_pos, avg_entry = ticker_state.get(ticker_key, (0.0, 0.0))
        delta = size if action == "buy" else -size
        new_pos = prev_pos + delta

        realized_delta = 0.0

        if prev_pos == 0:
            avg_entry = price
        elif (prev_pos > 0 and delta > 0) or (prev_pos < 0 and delta < 0):
            avg_entry = (avg_entry * abs(prev_pos) + price * abs(delta)) / abs(new_pos)
        else:
            closed = min(abs(delta), abs(prev_pos))
            if prev_pos > 0:
                realized_delta = (price - avg_entry) * closed
            else:
                realized_delta = (avg_entry - price) * closed
            if new_pos == 0:
                avg_entry = 0.0
            elif (prev_pos > 0 and new_pos < 0) or (prev_pos < 0 and new_pos > 0):
                avg_entry = price

        ticker_state[ticker_key] = (new_pos, avg_entry)
        pos = new_pos
        cum_realized += realized_delta
        cum_fees += fee
        cum_pnl = cum_realized - cum_fees

        gross_cost = price * size
        if action == "buy":
            net_cost = gross_cost + fee
        else:
            net_cost = -(gross_cost - fee)

        coid = str(f.get("client_order_id", ""))
        if coid.startswith("init_") or coid == "init":
            tag = "init"
        elif coid.startswith("phase3t_"):
            tag = "phase3:time"
        elif coid.startswith("phase3d_"):
            tag = "phase3:drift"
        elif coid.startswith("phase3_"):
            tag = "phase3"
        elif coid.startswith("flat_"):
            tag = "flat"
        else:
            tag = ""

        is_taker = bool(f.get("is_taker", 0))
        role = "TAKER" if is_taker else "MAKER"

        rows.append({
            "ts": ts,
            "ticker": f.get("ticker", ""),
            "action": action.upper(),
            "size": size,
            "price": price,
            "fee": fee,
            "gross": gross_cost,
            "net": net_cost,
            "pos": pos,
            "avg": avg_entry if pos != 0 else 0.0,
            "realized_delta": realized_delta,
            "cum_pnl": cum_pnl,
            "tag": tag,
            "role": role,
        })

    df = pd.DataFrame(rows)
    df["ts"] = to_ct(df["ts"]).dt.strftime("%m-%d %H:%M:%S")
    return df


# =============================================================================
# Main Window
# =============================================================================

class PositionManager(QMainWindow):

    # Two column sets — 15-min view inserts a Ticker column since one
    # file holds many markets.  Daily view doesn't need it (one event
    # per file → ticker is redundant given the strike already groups).
    COLUMNS_DAILY = [
        ("Time", "ts"),
        ("Action", "action"),
        ("Size", "size"),
        ("Price", "price"),
        ("Fee", "fee"),
        ("Gross", "gross"),
        ("Net", "net"),
        ("Pos", "pos"),
        ("Avg Entry", "avg"),
        ("Realized Δ", "realized_delta"),
        ("Cum P&L", "cum_pnl"),
        ("Tag", "tag"),
        ("Role", "role"),
    ]
    WIDTHS_DAILY = [140, 70, 60, 70, 60, 80, 90, 60, 80, 90, 90, 60, 60]

    COLUMNS_15M = [
        ("Time", "ts"),
        ("Ticker", "ticker"),
        ("Action", "action"),
        ("Size", "size"),
        ("Price", "price"),
        ("Fee", "fee"),
        ("Gross", "gross"),
        ("Net", "net"),
        ("Pos", "pos"),
        ("Avg Entry", "avg"),
        ("Realized Δ", "realized_delta"),
        ("Cum P&L", "cum_pnl"),
        ("Role", "role"),
    ]
    WIDTHS_15M = [140, 220, 70, 60, 70, 60, 80, 90, 60, 80, 90, 90, 60]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Position Manager")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Mode toggle ---
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Daily 5pm", "15-Min"])
        self.mode_combo.setMinimumWidth(140)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addSpacing(40)

        # Daily-mode pickers
        self.daily_event_label = QLabel("Event:")
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(220)
        self.event_combo.currentTextChanged.connect(self._on_event_changed)
        self.daily_strike_label = QLabel("Strike:")
        self.strike_combo = QComboBox()
        self.strike_combo.setMinimumWidth(140)
        self.strike_combo.currentTextChanged.connect(self._refresh_table)
        for w in (self.daily_event_label, self.event_combo,
                  self.daily_strike_label, self.strike_combo):
            mode_row.addWidget(w)

        # 15-min mode pickers (Series + Day + Market)
        self.m15_series_label = QLabel("Series:")
        self.m15_series_combo = QComboBox()
        self.m15_series_combo.setMinimumWidth(120)
        self.m15_series_combo.currentTextChanged.connect(self._on_15m_series_changed)
        self.m15_day_label = QLabel("Day:")
        self.m15_day_combo = QComboBox()
        self.m15_day_combo.setMinimumWidth(100)
        self.m15_day_combo.currentTextChanged.connect(self._refresh_table)
        self.m15_market_label = QLabel("Market:")
        self.m15_market_combo = QComboBox()
        self.m15_market_combo.setMinimumWidth(240)
        self.m15_market_combo.currentTextChanged.connect(self._refresh_table)
        for w in (self.m15_series_label, self.m15_series_combo,
                  self.m15_day_label, self.m15_day_combo,
                  self.m15_market_label, self.m15_market_combo):
            mode_row.addWidget(w)

        mode_row.addStretch()
        layout.addLayout(mode_row)

        # --- Summary panel ---
        self.summary = QLabel("")
        self.summary.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setStyleSheet(
            "color:#c8cdd5;background:#141923;"
            "padding:8px 12px;border-radius:4px;"
        )
        layout.addWidget(self.summary)

        # --- Trade table (columns rebuilt on mode change) ---
        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # Active column set — updated by _on_mode_changed
        self._columns = self.COLUMNS_DAILY
        self._widths = self.WIDTHS_DAILY
        self._apply_table_columns()

        self.setStyleSheet(
            "QMainWindow{background:#0b0f19;}"
            "QWidget{background:#0b0f19;color:#c8cdd5;font-family:Courier;}"
            "QLabel{color:#c8cdd5;font-size:12px;}"
            "QComboBox{background:#141923;color:#c8cdd5;"
            "border:1px solid #1e2736;border-radius:3px;padding:4px;}"
            "QComboBox QAbstractItemView{background:#141923;color:#c8cdd5;"
            "selection-background-color:#1e2736;}"
            "QTableWidget{background:#0b0f19;color:#c8cdd5;"
            "gridline-color:#1e2736;border:1px solid #1e2736;}"
            "QHeaderView::section{background:#1e2736;color:#facc15;"
            "padding:6px;border:1px solid #2d3a4d;font-weight:bold;}"
        )

        # Initialize in Daily mode
        self._on_mode_changed(0)

        # Auto-refresh every 30s
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(30_000)

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def _is_15m_mode(self) -> bool:
        return self.mode_combo.currentIndex() == 1

    def _on_mode_changed(self, index: int):
        is_15m = (index == 1)
        # Show/hide daily-mode widgets
        for w in (self.daily_event_label, self.event_combo,
                  self.daily_strike_label, self.strike_combo):
            w.setVisible(not is_15m)
        # Show/hide 15-min widgets
        for w in (self.m15_series_label, self.m15_series_combo,
                  self.m15_day_label, self.m15_day_combo,
                  self.m15_market_label, self.m15_market_combo):
            w.setVisible(is_15m)
        # Swap column set + repopulate the picker(s) for the new mode.
        if is_15m:
            self._columns = self.COLUMNS_15M
            self._widths = self.WIDTHS_15M
            self._populate_15m_series()
        else:
            self._columns = self.COLUMNS_DAILY
            self._widths = self.WIDTHS_DAILY
            self._populate_daily_events()
        self._apply_table_columns()

    def _apply_table_columns(self):
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels([c[0] for c in self._columns])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for i, w in enumerate(self._widths):
            self.table.setColumnWidth(i, w)

    # ------------------------------------------------------------------
    # Daily 5pm pickers
    # ------------------------------------------------------------------

    def _populate_daily_events(self):
        events = list_daily_events()
        current = self.event_combo.currentText()
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        self.event_combo.addItems(events)
        if current and current in events:
            self.event_combo.setCurrentText(current)
        elif events:
            self.event_combo.setCurrentIndex(len(events) - 1)
        self.event_combo.blockSignals(False)
        self._on_event_changed(self.event_combo.currentText())

    def _on_event_changed(self, event_ticker: str):
        fills = load_fills(event_ticker)
        counts = (fills.groupby("strike").size().to_dict()
                  if not fills.empty else {})
        strikes = sorted(counts.keys())

        current_data = self.strike_combo.currentData()
        self.strike_combo.blockSignals(True)
        self.strike_combo.clear()
        for s in strikes:
            n = int(counts.get(s, 0))
            label = f"${s:,.0f} ({n})"
            self.strike_combo.addItem(label, s)
        if current_data is not None:
            for i in range(self.strike_combo.count()):
                if self.strike_combo.itemData(i) == current_data:
                    self.strike_combo.setCurrentIndex(i)
                    break
        self.strike_combo.blockSignals(False)
        self._refresh_table()

    # ------------------------------------------------------------------
    # 15-min pickers
    # ------------------------------------------------------------------

    def _populate_15m_series(self):
        """List distinct series (e.g. KXBTC15M) across all 15-min files."""
        files = list_15m_files()
        series_set = set()
        for stem in files:
            s, _ = split_15m_stem(stem)
            if s:
                series_set.add(s)
        series_list = sorted(series_set)
        current = self.m15_series_combo.currentText()
        self.m15_series_combo.blockSignals(True)
        self.m15_series_combo.clear()
        self.m15_series_combo.addItems(series_list)
        if current and current in series_list:
            self.m15_series_combo.setCurrentText(current)
        elif series_list:
            self.m15_series_combo.setCurrentIndex(0)
        self.m15_series_combo.blockSignals(False)
        self._on_15m_series_changed(self.m15_series_combo.currentText())

    def _on_15m_series_changed(self, series: str):
        """Populate the day combo for the selected series.

        Sort chronologically using parse_day_label so April / August /
        December don't get re-ordered alphabetically.  Raw label
        (e.g. `26MAY14`) is stored as item data so the file lookup
        downstream is unchanged; only the display text is humanized."""
        files = list_15m_files()
        days = []
        for stem in files:
            s, d = split_15m_stem(stem)
            if s == series and d:
                days.append(d)
        # Sort by parsed date if available; unparseable labels go last.
        from datetime import date
        SENTINEL = date(1900, 1, 1)
        days = sorted(set(days),
                      key=lambda d: (parse_day_label(d) or SENTINEL, d))

        current_data = self.m15_day_combo.currentData()
        self.m15_day_combo.blockSignals(True)
        self.m15_day_combo.clear()
        for d in days:
            self.m15_day_combo.addItem(display_day_label(d), d)
        # Preserve selection by raw label, else default to the newest.
        if current_data:
            for i in range(self.m15_day_combo.count()):
                if self.m15_day_combo.itemData(i) == current_data:
                    self.m15_day_combo.setCurrentIndex(i)
                    break
            else:
                if self.m15_day_combo.count():
                    self.m15_day_combo.setCurrentIndex(
                        self.m15_day_combo.count() - 1)
        elif self.m15_day_combo.count():
            self.m15_day_combo.setCurrentIndex(
                self.m15_day_combo.count() - 1)
        self.m15_day_combo.blockSignals(False)
        self._populate_15m_markets()

    def _populate_15m_markets(self):
        """Populate the ticker selector for the chosen series+day file."""
        series = self.m15_series_combo.currentText()
        day = self.m15_day_combo.currentText()
        if not series or not day:
            self.m15_market_combo.blockSignals(True)
            self.m15_market_combo.clear()
            self.m15_market_combo.blockSignals(False)
            self._refresh_table()
            return
        stem = f"{series}-{day}"
        fills = load_fills(stem)
        counts = (fills.groupby("ticker").size().to_dict()
                  if not fills.empty else {})
        tickers = sorted(counts.keys())

        current_data = self.m15_market_combo.currentData()
        self.m15_market_combo.blockSignals(True)
        self.m15_market_combo.clear()
        # "All Today" rolls every fill in the file into one chronological view.
        self.m15_market_combo.addItem(
            f"All ({sum(counts.values())} fills)", None)
        for t in tickers:
            self.m15_market_combo.addItem(f"{t} ({counts[t]})", t)
        # Preserve selection if it still exists
        if current_data is not None:
            for i in range(self.m15_market_combo.count()):
                if self.m15_market_combo.itemData(i) == current_data:
                    self.m15_market_combo.setCurrentIndex(i)
                    break
        self.m15_market_combo.blockSignals(False)
        self._refresh_table()

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------

    def _auto_refresh(self):
        # Repopulate the active mode's pickers so newly-arrived files
        # show up.  Selection preserved.
        if self._is_15m_mode():
            self._populate_15m_series()
        else:
            self._populate_daily_events()

    # ------------------------------------------------------------------
    # Table refresh
    # ------------------------------------------------------------------

    def _refresh_table(self, *_):
        if self._is_15m_mode():
            series = self.m15_series_combo.currentText()
            day = self.m15_day_combo.currentText()
            if not series or not day:
                self.table.setRowCount(0)
                self.summary.setText("")
                return
            stem = f"{series}-{day}"
            fills = load_fills(stem)
            if fills.empty:
                self.table.setRowCount(0)
                self.summary.setText("No fills recorded.")
                return
            market = self.m15_market_combo.currentData()
            if market is not None:
                fills = fills[fills["ticker"] == market]
                if fills.empty:
                    self.table.setRowCount(0)
                    self.summary.setText(f"No fills for {market}")
                    return
            df = compute_trade_view(fills)
            self._update_summary_15m(df, series, day, market)
            self._fill_table(df.iloc[::-1].reset_index(drop=True))
        else:
            event = self.event_combo.currentText()
            strike_data = self.strike_combo.currentData()
            if not event or strike_data is None:
                self.table.setRowCount(0)
                self.summary.setText("")
                return
            fills = load_fills(event)
            if fills.empty:
                self.table.setRowCount(0)
                self.summary.setText("No fills recorded.")
                return
            fills = fills[fills["strike"] == strike_data]
            if fills.empty:
                self.table.setRowCount(0)
                self.summary.setText(f"No fills for ${strike_data:,.0f}")
                return
            df = compute_trade_view(fills)
            self._update_summary_daily(df, strike_data)
            self._fill_table(df.iloc[::-1].reset_index(drop=True))

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    def _fill_table(self, df: pd.DataFrame):
        self.table.setRowCount(len(df))
        for row_idx, (_, r) in enumerate(df.iterrows()):
            for col_idx, (_, key) in enumerate(self._columns):
                v = r[key] if key in r else ""
                if isinstance(v, float):
                    if key in ("size", "pos"):
                        text = f"{v:.0f}"
                    elif key in ("fee", "gross", "net",
                                 "realized_delta", "cum_pnl"):
                        text = (f"${v:+.2f}"
                                if key in ("net", "realized_delta", "cum_pnl")
                                else f"${v:.2f}")
                    else:
                        text = f"${v:.3f}"
                else:
                    text = str(v)
                item = QTableWidgetItem(text)
                if key == "action":
                    item.setForeground(QColor("#22c55e" if v == "BUY"
                                              else "#ef4444"))
                    f = item.font(); f.setBold(True); item.setFont(f)
                elif key == "cum_pnl":
                    item.setForeground(QColor("#22c55e" if v >= 0
                                              else "#ef4444"))
                elif key == "realized_delta":
                    if v > 0:
                        item.setForeground(QColor("#22c55e"))
                    elif v < 0:
                        item.setForeground(QColor("#ef4444"))
                    else:
                        item.setForeground(QColor("#5a6270"))
                elif key == "fee":
                    item.setForeground(QColor("#f59e0b"))
                elif key == "tag":
                    if v == "init":
                        item.setForeground(QColor("#facc15"))
                    elif v == "flat":
                        item.setForeground(QColor("#ff8c00"))
                    elif isinstance(v, str) and v.startswith("phase3"):
                        item.setForeground(QColor("#ef4444"))
                        f = item.font(); f.setBold(True); item.setFont(f)
                    else:
                        item.setForeground(QColor("#5a6270"))
                elif key == "role":
                    if v == "TAKER":
                        item.setForeground(QColor("#ef4444"))
                    else:
                        item.setForeground(QColor("#22c55e"))
                elif key == "ticker":
                    item.setForeground(QColor("#94a3b8"))
                self.table.setItem(row_idx, col_idx, item)

    # ------------------------------------------------------------------
    # Summary panel (one variant per mode)
    # ------------------------------------------------------------------

    def _summary_stats(self, df: pd.DataFrame) -> dict:
        return {
            "n": len(df),
            "n_buys": int((df["action"] == "BUY").sum()),
            "n_sells": int((df["action"] == "SELL").sum()),
            "n_makers": int((df["role"] == "MAKER").sum()),
            "n_takers": int((df["role"] == "TAKER").sum()),
            "size": float(df["size"].sum()),
            "fees": float(df["fee"].sum()),
            "realized": float(df["realized_delta"].sum()),
            "final_pnl": float(df["cum_pnl"].iloc[-1]) if not df.empty else 0.0,
            "final_pos": int(df["pos"].iloc[-1]) if not df.empty else 0,
            "final_avg": float(df["avg"].iloc[-1]) if not df.empty else 0.0,
        }

    def _update_summary_daily(self, df: pd.DataFrame, strike: float):
        s = self._summary_stats(df)
        pnl_color = "#22c55e" if s["final_pnl"] >= 0 else "#ef4444"
        pos_color = ("#22c55e" if s["final_pos"] > 0
                     else "#ef4444" if s["final_pos"] < 0 else "#5a6270")
        avg_text = f"${s['final_avg']:.3f}" if s["final_pos"] != 0 else "Flat"
        self.summary.setText(
            f"<b style='color:#facc15'>${strike:,.0f}</b> &nbsp;&nbsp;"
            f"Trades: {s['n']} ({s['n_buys']}B / {s['n_sells']}S) &nbsp;&nbsp;"
            f"<span style='color:#22c55e'>{s['n_makers']}M</span> / "
            f"<span style='color:#ef4444'>{s['n_takers']}T</span> &nbsp;&nbsp;"
            f"Size: {s['size']:.0f} &nbsp;&nbsp;"
            f"Pos: <span style='color:{pos_color}'>{s['final_pos']}</span> &nbsp;&nbsp;"
            f"Avg: {avg_text} &nbsp;&nbsp;"
            f"Realized: ${s['realized']:+.2f} &nbsp;&nbsp;"
            f"Fees: <span style='color:#f59e0b'>${s['fees']:.2f}</span> &nbsp;&nbsp;"
            f"Net P&amp;L: <span style='color:{pnl_color};font-size:14px'>"
            f"${s['final_pnl']:+.2f}</span>"
        )

    def _update_summary_15m(self, df: pd.DataFrame, series: str, day: str,
                            market: str | None):
        s = self._summary_stats(df)
        pnl_color = "#22c55e" if s["final_pnl"] >= 0 else "#ef4444"
        scope = f"{series} • {day}"
        if market:
            scope += f" • {market}"
        else:
            scope += " • All markets"
        self.summary.setText(
            f"<b style='color:#facc15'>{scope}</b> &nbsp;&nbsp;"
            f"Trades: {s['n']} ({s['n_buys']}B / {s['n_sells']}S) &nbsp;&nbsp;"
            f"<span style='color:#22c55e'>{s['n_makers']}M</span> / "
            f"<span style='color:#ef4444'>{s['n_takers']}T</span> &nbsp;&nbsp;"
            f"Size: {s['size']:.0f} &nbsp;&nbsp;"
            f"Realized: ${s['realized']:+.2f} &nbsp;&nbsp;"
            f"Fees: <span style='color:#f59e0b'>${s['fees']:.2f}</span> &nbsp;&nbsp;"
            f"Net P&amp;L: <span style='color:{pnl_color};font-size:14px'>"
            f"${s['final_pnl']:+.2f}</span>"
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Position Manager")
    try:
        from Foundation import NSBundle  # type: ignore
        bundle = NSBundle.mainBundle()
        if bundle:
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info is not None:
                info["CFBundleName"] = "Position Manager"
                info["CFBundleDisplayName"] = "Position Manager"
    except Exception:
        pass
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
        try:
            from AppKit import NSApplication, NSImage  # type: ignore
            ns_img = NSImage.alloc().initByReferencingFile_(str(ICON_PATH))
            NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
        except Exception:
            pass
    win = PositionManager()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
