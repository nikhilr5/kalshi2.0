"""UI — PyQt6 window for the weather-floor app.

Dropdowns for city / side (High|Low) / date, a size spinbox, an ARM toggle, the
current MADIS running high-or-low for the day, and a live bucket table (range,
strike, dead?, yes bid/ask, our position, time-to-expiration). All times ET.

The window is passive: it emits high-level signals (seriesChanged / dateChanged /
armToggled / sizeChanged) that app.py wires to the controller, and exposes
refresh(snapshot) + populate_dates() + log() for the controller to drive.
"""
import collections
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QPlainTextEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

_COLS = ["Bucket", "Range", "Dead?", "Yes Bid", "Yes Ask", "Pos"]


class TradeWindow(QMainWindow):
    seriesChanged = pyqtSignal()      # city or side changed -> reload dates
    dateChanged = pyqtSignal()        # date changed -> apply selection
    armToggled = pyqtSignal(bool)
    sizeChanged = pyqtSignal(int)
    maxPosChanged = pyqtSignal(int)
    cooldownChanged = pyqtSignal(float)
    cushionChanged = pyqtSignal(float)

    def __init__(self, cities: list[str]):
        super().__init__()
        self.setWindowTitle("Tundra — Weather Floor")
        self.resize(900, 600)
        self._loading = False
        # log buffer: worker threads (WS/MADIS/OSM) append here; only the main
        # thread (refresh) touches the widget. Calling Qt widgets off-thread segfaults.
        self._logq = collections.deque(maxlen=1000)
        self._loglock = threading.Lock()
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # ---- controls row ----
        ctrl = QHBoxLayout()
        self.cityCombo = QComboBox(); self.cityCombo.addItems(cities)
        self.sideCombo = QComboBox(); self.sideCombo.addItems(["Low", "High"])
        self.dateCombo = QComboBox()
        self.sizeSpin = QSpinBox(); self.sizeSpin.setRange(1, 500); self.sizeSpin.setValue(1)
        self.maxSpin = QSpinBox(); self.maxSpin.setRange(0, 5000); self.maxSpin.setValue(10)
        self.coolSpin = QDoubleSpinBox(); self.coolSpin.setRange(0, 120); self.coolSpin.setSingleStep(0.5)
        self.coolSpin.setValue(2.0); self.coolSpin.setSuffix(" s")
        self.cushSpin = QDoubleSpinBox(); self.cushSpin.setRange(0, 5); self.cushSpin.setSingleStep(0.5)
        self.cushSpin.setValue(2.0); self.cushSpin.setSuffix(" °F")
        self.armCheck = QCheckBox("ARM (sell dead buckets)")
        for lab, w in (("City", self.cityCombo), ("Side", self.sideCombo),
                       ("Date", self.dateCombo), ("Size", self.sizeSpin),
                       ("Max pos", self.maxSpin), ("Cooldown", self.coolSpin),
                       ("Cushion", self.cushSpin)):
            ctrl.addWidget(QLabel(lab)); ctrl.addWidget(w)
        ctrl.addWidget(self.armCheck)
        ctrl.addStretch(1)
        v.addLayout(ctrl)

        # ---- current extreme banner ----
        self.extremeLabel = QLabel("—")
        self.extremeLabel.setStyleSheet("font-size: 15px; font-weight: bold; padding: 6px;")
        v.addWidget(self.extremeLabel)

        # ---- bucket table ----
        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        v.addWidget(self.table, 1)

        # ---- log ----
        self.logBox = QPlainTextEdit(); self.logBox.setReadOnly(True)
        self.logBox.setMaximumBlockCount(500); self.logBox.setFixedHeight(140)
        v.addWidget(self.logBox)

        # ---- wiring ----
        self.cityCombo.currentTextChanged.connect(lambda *_: self.seriesChanged.emit())
        self.sideCombo.currentTextChanged.connect(lambda *_: self.seriesChanged.emit())
        self.dateCombo.currentTextChanged.connect(self._on_date)
        self.armCheck.toggled.connect(self.armToggled.emit)
        self.sizeSpin.valueChanged.connect(self.sizeChanged.emit)
        self.maxSpin.valueChanged.connect(self.maxPosChanged.emit)
        self.coolSpin.valueChanged.connect(self.cooldownChanged.emit)
        self.cushSpin.valueChanged.connect(self.cushionChanged.emit)

    # ---- selection accessors ----
    def selection(self) -> tuple[str, str, str]:
        return (self.cityCombo.currentText(),
                self.sideCombo.currentText().lower(),
                self.dateCombo.currentText())

    def populate_dates(self, dates: list[str], default: str | None = None):
        self._loading = True
        self.dateCombo.clear()
        self.dateCombo.addItems(dates)
        if default and default in dates:
            self.dateCombo.setCurrentText(default)      # default to today's climate day
        self._loading = False

    def _on_date(self, *_):
        if not self._loading and self.dateCombo.currentText():
            self.dateChanged.emit()

    # ---- driven by controller ----
    def log(self, msg: str):
        """Thread-safe: just buffer. The widget is written only in _drain_log()
        on the main thread (via refresh)."""
        with self._loglock:
            self._logq.append(msg)

    def _drain_log(self):
        with self._loglock:
            msgs = list(self._logq)
            self._logq.clear()
        for m in msgs:
            self.logBox.appendPlainText(m)

    def refresh(self, snap: dict):
        self._drain_log()
        side = (snap.get("side") or "low")
        word = "High" if side == "high" else "Low"
        ex = snap.get("extreme")
        age = snap.get("asof_age")
        age_str, color = _age(age)
        head = f"{snap.get('city','—')} {snap.get('event_day','')}"
        if ex is None:
            self.extremeLabel.setText(f"{head} — running {word}: no MADIS obs yet")
        else:
            self.extremeLabel.setText(
                f"{head} — running {word}: {ex:.0f}°F   "
                f"obs {snap.get('asof_et','—')} ({age_str})   ·   "
                f"expires in {snap.get('tte','—')}   ·   "
                f"short {snap.get('total_short',0)}/{snap.get('max_position',0)}")
        self.extremeLabel.setStyleSheet(
            f"font-size: 14px; font-weight: bold; padding: 6px; color: {color};")

        rows = snap.get("buckets", [])
        self.table.setRowCount(len(rows))
        for r, b in enumerate(rows):
            tail = b["ticker"].split("-")[-1]
            vals = [tail, b.get("sub_title", ""),
                    "DEAD" if b["dead"] else "", _px(b["yes_bid"]), _px(b["yes_ask"]),
                    str(b.get("pos", 0))]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                if c >= 1:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if b["dead"]:
                    it.setBackground(QColor(70, 30, 30))      # dead = red-ish
                if c == 5 and b.get("pos", 0) < 0:
                    it.setForeground(QColor(120, 220, 120))   # short = green
                self.table.setItem(r, c, it)


def _px(p) -> str:
    return f"{p:.2f}" if p and p > 0 else "—"


def _age(secs) -> tuple[str, str]:
    """(human age string, color) for MADIS obs freshness. Stale feed = stale low."""
    if secs is None:
        return "no obs", "#888"
    m, s = divmod(int(secs), 60)
    txt = f"{m}m{s:02d}s ago" if m else f"{s}s ago"
    if secs <= 8 * 60:
        return txt, "#5fd35f"      # fresh
    if secs <= 15 * 60:
        return txt, "#d8c84a"      # getting old
    return txt + " — STALE", "#e05a5a"
