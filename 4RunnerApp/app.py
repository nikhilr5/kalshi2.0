"""
Kalshi Bracket Trader — PyQt6 Desktop App

Features:
    - Live YES bid/ask via Kalshi websocket
    - Live BTC/USD via Coinbase websocket
    - Portfolio balance display
    - Sell YES orders at bid + edge
    - Balance check before placing orders
    - Auto-cancel resting orders X hours before expiration
    - Email notification on fills
    - Graceful Ctrl+C shutdown
"""

import sys
import signal
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox,
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from kalshi_api import KalshiAPI
from order_manager import OrderManager
from market_discovery import discover_weekly_events, parse_strike
from ws_feed import KalshiWsFeed
from btc_price_feed import BtcPriceFeed
from PyQt6.QtGui import QIcon


# =============================================================================
# Background Workers
# =============================================================================

class DiscoverWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api, series, weeks_ahead):
        super().__init__()
        self.api = api
        self.series = series
        self.weeks_ahead = weeks_ahead

    def run(self):
        try:
            events = discover_weekly_events(self.api, self.series, self.weeks_ahead)
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
# Main Application
# =============================================================================

class TradingApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kalshi Bracket Trader")
        self.setMinimumSize(1500, 700)

        self.setWindowTitle("4Runner Trader")
        self.setWindowIcon(QIcon("icon.png"))

        self.api = KalshiAPI()
        self.order_mgr = OrderManager(self.api, default_quantity=50)
        self.ws_feed = None
        self.btc_feed = None
        self.events = []
        self.current_event = None
        self.btc_price = 0.0

        self._discover_worker = None
        self._portfolio_worker = None
        self._order_worker = None
        self._cancel_worker = None

        self._build_ui()

        # Start BTC price feed
        self.btc_feed = BtcPriceFeed(self._on_btc_price)
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

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        top = QHBoxLayout()

        top.addWidget(QLabel("Event:"))
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(350)
        self.event_combo.currentIndexChanged.connect(self._on_event_changed)
        top.addWidget(self.event_combo)

        top.addSpacing(15)
        top.addWidget(QLabel("Edge:"))
        self.edge_input = QLineEdit("0.14")
        self.edge_input.setMaximumWidth(70)
        self.edge_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edge_input.textChanged.connect(self._mark_dirty)
        top.addWidget(self.edge_input)

        top.addSpacing(15)
        top.addWidget(QLabel("Size:"))
        self.size_input = QLineEdit("50")
        self.size_input.setMaximumWidth(50)
        self.size_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_input.textChanged.connect(self._on_size_changed)
        top.addWidget(self.size_input)

        top.addSpacing(15)
        top.addWidget(QLabel("Auto-cancel (hrs):"))
        self.auto_cancel_input = QLineEdit("2")
        self.auto_cancel_input.setMaximumWidth(40)
        self.auto_cancel_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.auto_cancel_input)

        top.addSpacing(15)
        top.addWidget(QLabel("BTC:"))
        self.btc_label = QLabel("--")
        self.btc_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.btc_label.setStyleSheet("color: #22c55e;")
        self.btc_label.setMinimumWidth(110)
        top.addWidget(self.btc_label)

        top.addSpacing(15)
        top.addWidget(QLabel("Balance:"))
        self.balance_label = QLabel("--")
        self.balance_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.balance_label.setStyleSheet("color: #f59e0b;")
        self.balance_label.setMinimumWidth(150)
        top.addWidget(self.balance_label)

        top.addSpacing(15)
        top.addWidget(QLabel("Expires:"))
        self.countdown_label = QLabel("--:--:--")
        self.countdown_label.setFont(QFont("Courier", 12, QFont.Weight.Bold))
        self.countdown_label.setStyleSheet("color: #f59e0b;")
        top.addWidget(self.countdown_label)

        top.addStretch()

        self.cancel_all_btn = QPushButton("Cancel All Orders")
        self.cancel_all_btn.setStyleSheet(
            "QPushButton{background:#ef4444;color:white;padding:8px 16px;"
            "border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#dc2626;}"
        )
        self.cancel_all_btn.clicked.connect(self._cancel_all_orders)
        top.addWidget(self.cancel_all_btn)

        layout.addLayout(top)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Contract", "Yes Bid", "Yes Ask", "Place Order",
            "Position ($)", "Open Qty ($)", "Open Level", "Status"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 250)
        header.resizeSection(1, 70)
        header.resizeSection(2, 70)
        header.resizeSection(3, 150)
        header.resizeSection(4, 90)
        header.resizeSection(5, 90)
        header.resizeSection(6, 80)
        header.resizeSection(7, 70)
        header.setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("color:#5a6270;font-size:11px;")
        layout.addWidget(self.status_label)

    # =========================================================================
    # BTC Price
    # =========================================================================

    def _on_btc_price(self, price: float):
        self.btc_price = price
        self._btc_dirty = True

    def _flush_btc_price(self):
        if self._btc_dirty:
            self._btc_dirty = False
            if self.btc_price > 0:
                self.btc_label.setText(f"${self.btc_price:,.2f}")

    # =========================================================================
    # Event Discovery
    # =========================================================================

    def _discover_events(self):
        self.status_label.setText("Discovering events...")
        self._discover_worker = DiscoverWorker(self.api, "KXBTC", 3)
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

    def _on_ws_update(self, ticker, yes_bid, yes_ask):
        self.order_mgr.update_book(ticker, yes_bid, yes_ask)
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
            return
        self._portfolio_worker = PortfolioWorker(self.order_mgr)
        self._portfolio_worker.finished.connect(self._on_portfolio_done)
        self._portfolio_worker.start()

    def _on_portfolio_done(self):
        self._table_dirty = True
        bal = self.order_mgr.balance_dollars
        pv = self.order_mgr.portfolio_value_dollars
        total = bal + pv
        self.balance_label.setText(f"${total:,.2f}")
        self.balance_label.setToolTip(
            f"Available cash: ${bal:,.2f}\nPositions: ${pv:,.2f}\nTotal: ${total:,.2f}"
        )

    # =========================================================================
    # Auto-Cancel
    # =========================================================================

    def _check_auto_cancel(self):
        if not self.current_event:
            return
        close_str = self.current_event.get("close_time", "")
        if not close_str:
            return
        try:
            hours = float(self.auto_cancel_input.text())
        except ValueError:
            return
        try:
            close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            now = datetime.now(close.tzinfo)
            remaining_hours = (close - now).total_seconds() / 3600.0

            if 0 < remaining_hours <= hours:
                has_orders = any(s.order_id for s in self.order_mgr.contracts.values())
                if has_orders:
                    self.status_label.setText(
                        f"Auto-cancelling: {remaining_hours:.1f}h to expiry"
                    )
                    self._cancel_worker = OrderWorker(self.order_mgr.cancel_all_orders)
                    self._cancel_worker.finished.connect(
                        lambda r: self.status_label.setText(f"Auto-cancelled: {r}")
                    )
                    self._cancel_worker.start()
        except Exception:
            pass

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
            for col in (1, 2):
                self.table.setItem(row, col, QTableWidgetItem("--"))
            btn = QPushButton("--")
            btn.setStyleSheet(
                "QPushButton{background:#1e2736;color:#ef4444;padding:4px 6px;"
                "border:1px solid #ef4444;border-radius:3px;"
                "font-weight:bold;font-size:10px;}"
                "QPushButton:hover{background:#ef4444;color:white;}"
            )
            btn.clicked.connect(lambda _, t=state.ticker: self._place_order(t))
            self.table.setCellWidget(row, 3, btn)
            for col in range(4, 8):
                self.table.setItem(row, col, QTableWidgetItem("--"))

    def _update_table(self):
        edge = self._get_edge()
        contracts = list(self.order_mgr.contracts.values())
        for row, state in enumerate(contracts):
            if row >= self.table.rowCount():
                break

            item = self.table.item(row, 1)
            if item:
                if state.best_bid > 0:
                    item.setText(f"${state.best_bid:.2f}")
                    item.setForeground(QColor("#22c55e"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            item = self.table.item(row, 2)
            if item:
                if state.best_ask > 0:
                    item.setText(f"${state.best_ask:.2f}")
                    item.setForeground(QColor("#ef4444"))
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            btn = self.table.cellWidget(row, 3)
            if btn and isinstance(btn, QPushButton):
                if state.best_bid > 0:
                    sell_price = round(state.best_bid + edge, 2)
                else:
                    sell_price = round(edge, 2)
                btn.setText(f"Sell YES @ ${sell_price:.2f}")

            item = self.table.item(row, 4)
            if item:
                if state.position_dollars != 0:
                    item.setText(f"${state.position_dollars:.2f}")
                    item.setForeground(
                        QColor("#22c55e") if state.position_dollars > 0
                        else QColor("#ef4444")
                    )
                else:
                    item.setText("--")
                    item.setForeground(QColor("#5a6270"))

            item = self.table.item(row, 5)
            if item:
                item.setText(f"${state.open_quantity_dollars:.2f}" if state.open_quantity_dollars > 0 else "--")

            item = self.table.item(row, 6)
            if item:
                item.setText(f"${state.open_level:.2f}" if state.open_level > 0 else "--")

            item = self.table.item(row, 7)
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
    # Order Placement
    # =========================================================================

    def _place_order(self, ticker: str):
        edge = self._get_edge()
        state = self.order_mgr.contracts.get(ticker)

        if not state:
            QMessageBox.warning(self, "Error", f"Contract not found: {ticker}")
            return

        # Calculate sell price
        if state.best_bid > 0:
            sell_price = round(state.best_bid + edge, 2)
        else:
            sell_price = round(edge, 2)

        if sell_price <= 0 or sell_price >= 1.0:
            QMessageBox.warning(self, "Error", f"Invalid price: ${sell_price:.2f}")
            return

        qty = self.order_mgr.default_quantity
        available = self.order_mgr.balance_dollars

        # Collateral = max loss per contract × qty
        # When selling YES at $X, max loss = $1.00 - $X (if contract settles YES)
        max_loss_per = round(1.0 - sell_price, 2)
        collateral = max_loss_per * qty

        # Profit if contract settles NO (the usual case for OTM brackets)
        profit_if_no = sell_price * qty

        # Check balance
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

        # Confirmation with clear cost/reward breakdown
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
    # Helpers
    # =========================================================================

    def _get_edge(self) -> float:
        try:
            return float(self.edge_input.text())
        except ValueError:
            return 0.14

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
    app = QApplication(sys.argv)
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
    app.setWindowIcon(QIcon("icon.png"))

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