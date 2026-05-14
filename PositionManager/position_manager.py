"""
Position manager — PyQt app for viewing all trades for an event/strike with
PnL breakdown.

Reads from analysis/backtesting/data/<event>.db (per-event fill files).

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


def list_events():
    rows = []
    for db_file in DATA_DIR.glob("*.db"):
        if db_file.name == "recorder.db":
            continue
        rows.append(db_file.stem)
    return sorted(rows)


def load_fills(event_ticker: str) -> pd.DataFrame:
    if not event_ticker:
        return pd.DataFrame()
    db_file = event_db_path(event_ticker)
    if not db_file.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_file))
    try:
        df = pd.read_sql("SELECT * FROM fills ORDER BY ts", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def list_strikes(event_ticker: str):
    df = load_fills(event_ticker)
    if df.empty:
        return []
    return sorted(df["strike"].unique())


def compute_trade_view(fills: pd.DataFrame) -> pd.DataFrame:
    """Walk fills chronologically; compute position + cumulative P&L per row."""
    if fills.empty:
        return pd.DataFrame()

    fills = fills.sort_values("ts").reset_index(drop=True)
    rows = []
    pos = 0
    avg_entry = 0.0
    cum_realized = 0.0
    cum_fees = 0.0

    for _, f in fills.iterrows():
        action = f["action"]
        size = float(f["count"])
        price = float(f["price"])
        fee = float(f.get("fee") or 0.0)
        ts = f["ts"]

        delta = size if action == "buy" else -size
        prev_pos = pos
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
        if coid.startswith("init_"):
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

    COLUMNS = [
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
    COL_WIDTHS = [140, 70, 60, 70, 60, 80, 90, 60, 80, 90, 90, 60, 60]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Position Manager")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Top bar: filters ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Event:"))
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(220)
        self.event_combo.currentTextChanged.connect(self._on_event_changed)
        top.addWidget(self.event_combo)

        top.addSpacing(20)
        top.addWidget(QLabel("Strike:"))
        self.strike_combo = QComboBox()
        self.strike_combo.setMinimumWidth(140)
        self.strike_combo.currentTextChanged.connect(self._refresh_table)
        top.addWidget(self.strike_combo)

        top.addStretch()
        layout.addLayout(top)

        # --- Summary panel ---
        self.summary = QLabel("")
        self.summary.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setStyleSheet(
            "color:#c8cdd5;background:#141923;"
            "padding:8px 12px;border-radius:4px;"
        )
        layout.addWidget(self.summary)

        # --- Trade table ---
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in self.COLUMNS])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for i, w in enumerate(self.COL_WIDTHS):
            self.table.setColumnWidth(i, w)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # --- Stylesheet ---
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

        self._populate_events()

        # Auto-refresh every 30s
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(30_000)

    def _populate_events(self):
        events = list_events()
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
        # Count fills per strike
        fills = load_fills(event_ticker)
        if fills.empty:
            counts = {}
        else:
            counts = fills.groupby("strike").size().to_dict()
        strikes = sorted(counts.keys())

        # Preserve selection by data (strike value), not text
        current_data = self.strike_combo.currentData()
        self.strike_combo.blockSignals(True)
        self.strike_combo.clear()
        for s in strikes:
            n = int(counts.get(s, 0))
            label = f"${s:,.0f} ({n})"
            self.strike_combo.addItem(label, s)
        # Try to keep current selection
        if current_data is not None:
            for i in range(self.strike_combo.count()):
                if self.strike_combo.itemData(i) == current_data:
                    self.strike_combo.setCurrentIndex(i)
                    break
        self.strike_combo.blockSignals(False)
        self._refresh_table()

    def _auto_refresh(self):
        # Re-populate events (in case new ones appeared) but preserve selection
        self._populate_events()

    def _refresh_table(self, *_):
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
        # Compute summary from chronological view, then reverse for display
        # (most recent trade at the top)
        self._update_summary(df, strike_data)
        self._fill_table(df.iloc[::-1].reset_index(drop=True))

    def _fill_table(self, df: pd.DataFrame):
        self.table.setRowCount(len(df))
        for row_idx, (_, r) in enumerate(df.iterrows()):
            for col_idx, (_, key) in enumerate(self.COLUMNS):
                v = r[key]
                if isinstance(v, float):
                    if key in ("size", "pos"):
                        text = f"{v:.0f}"
                    elif key in ("fee", "gross", "net", "realized_delta", "cum_pnl"):
                        text = f"${v:+.2f}" if key in ("net", "realized_delta", "cum_pnl") else f"${v:.2f}"
                    else:
                        text = f"${v:.3f}"
                else:
                    text = str(v)
                item = QTableWidgetItem(text)
                # Color coding
                if key == "action":
                    item.setForeground(QColor("#22c55e" if v == "BUY" else "#ef4444"))
                    f = item.font(); f.setBold(True); item.setFont(f)
                elif key == "cum_pnl":
                    color = "#22c55e" if v >= 0 else "#ef4444"
                    item.setForeground(QColor(color))
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
                        item.setForeground(QColor("#ef4444"))  # red — forced cross
                        f = item.font(); f.setBold(True); item.setFont(f)
                    else:
                        item.setForeground(QColor("#5a6270"))
                elif key == "role":
                    if v == "TAKER":
                        item.setForeground(QColor("#ef4444"))  # red — paid spread
                    else:
                        item.setForeground(QColor("#22c55e"))  # green — captured spread
                self.table.setItem(row_idx, col_idx, item)

    def _update_summary(self, df: pd.DataFrame, strike: float):
        n_buys = int((df["action"] == "BUY").sum())
        n_sells = int((df["action"] == "SELL").sum())
        n_makers = int((df["role"] == "MAKER").sum())
        n_takers = int((df["role"] == "TAKER").sum())
        total_size = df["size"].sum()
        total_fees = df["fee"].sum()
        final_pnl = df["cum_pnl"].iloc[-1] if not df.empty else 0
        final_pos = int(df["pos"].iloc[-1]) if not df.empty else 0
        final_avg = df["avg"].iloc[-1] if not df.empty else 0
        realized_total = df["realized_delta"].sum()

        pnl_color = "#22c55e" if final_pnl >= 0 else "#ef4444"
        pos_color = "#22c55e" if final_pos > 0 else "#ef4444" if final_pos < 0 else "#5a6270"

        avg_text = f"${final_avg:.3f}" if final_pos != 0 else "Flat"

        self.summary.setText(
            f"<b style='color:#facc15'>${strike:,.0f}</b> &nbsp;&nbsp;"
            f"Trades: {len(df)} ({n_buys}B / {n_sells}S) &nbsp;&nbsp;"
            f"<span style='color:#22c55e'>{n_makers}M</span> / "
            f"<span style='color:#ef4444'>{n_takers}T</span> &nbsp;&nbsp;"
            f"Size: {total_size:.0f} &nbsp;&nbsp;"
            f"Pos: <span style='color:{pos_color}'>{final_pos}</span> &nbsp;&nbsp;"
            f"Avg: {avg_text} &nbsp;&nbsp;"
            f"Realized: ${realized_total:+.2f} &nbsp;&nbsp;"
            f"Fees: <span style='color:#f59e0b'>${total_fees:.2f}</span> &nbsp;&nbsp;"
            f"Net P&amp;L: <span style='color:{pnl_color};font-size:14px'>"
            f"${final_pnl:+.2f}</span>"
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Position Manager")
    # Override CFBundleName so the macOS Dock tooltip reads "Position
    # Manager" instead of "Python" (the default for raw .py runs).
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
        # On macOS, the Dock icon comes from NSApplication's icon image.
        # Push the icon to NSApp directly when PyObjC is available so the
        # Dock shows our icon instead of the generic Python rocket.
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
