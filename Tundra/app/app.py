"""app.py — main controller for the Tundra weather-floor app.

Wires the pieces (Aston-style separation of concerns):
    MADIS feed  ->  Strategy  ->  OSM  ->  KalshiAPI            (decide / sell)
    Kalshi WS   ->  Strategy.on_book (live BBO) + OSM.on_fill   (market data / fills)
    UI (PyQt6)  <-  QTimer reads Strategy.snapshot()            (display)

Run from Tundra/app/:   python3 app.py
Select City / Side (High|Low) / Date, watch the running high-or-low and which
buckets are DEAD, then tick ARM to start selling dead buckets into the bid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtCore import QTimer                     # noqa: E402
from PyQt6.QtWidgets import QApplication            # noqa: E402

from kalshi_api import KalshiAPI                    # noqa: E402
from kalshi_ws import KalshiWsFeed                  # noqa: E402
from madis_feed import MadisFeed                    # noqa: E402
from osm import OSM                                 # noqa: E402
from strategy import Strategy, CITIES               # noqa: E402
from ui import TradeWindow                          # noqa: E402


class Controller:
    """Translates UI selection into strategy/feed actions. No trading logic here."""

    def __init__(self, ui, osm, strategy, ws):
        self.ui = ui
        self.osm = osm
        self.strategy = strategy
        self.ws = ws
        self._ws_started = False

    def on_series_changed(self):
        city, side, _ = self.ui.selection()
        try:
            dates = self.strategy.list_event_days(city, side)
        except Exception as e:
            self.ui.log(f"[dates] load failed: {e}")
            return
        self.ui.populate_dates(dates, self.strategy.today_event_day(city))
        if dates:
            self.apply_selection()

    def on_date_changed(self):
        self.apply_selection()

    def apply_selection(self):
        city, side, date = self.ui.selection()
        if not date:
            return
        try:
            tickers = self.strategy.set_selection(city, side, date)
        except Exception as e:
            self.ui.log(f"[select] failed: {e}")
            return
        if not tickers:
            return
        if not self._ws_started:
            self.ws.start(tickers)
            self._ws_started = True
        else:
            self.ws.set_tickers(tickers)

    def refresh(self):
        try:
            self.ui.refresh(self.strategy.snapshot())
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    api = KalshiAPI()
    madis = MadisFeed()
    ui = TradeWindow(list(CITIES.keys()))
    osm = OSM(api, log=ui.log)
    strategy = Strategy(api, osm, madis, log=ui.log)
    ws = KalshiWsFeed(api, on_update=strategy.on_book, on_fill=osm.on_fill,
                      on_stale=lambda: ui.log("[WS] STALE — market data dropped"))
    ctrl = Controller(ui, osm, strategy, ws)

    ui.seriesChanged.connect(ctrl.on_series_changed)
    ui.dateChanged.connect(ctrl.on_date_changed)
    ui.armToggled.connect(osm.set_armed)
    ui.sizeChanged.connect(osm.set_size)
    ui.maxPosChanged.connect(osm.set_max_position)
    ui.cooldownChanged.connect(osm.set_cooldown)
    ui.cushionChanged.connect(strategy.set_cushion)

    timer = QTimer()
    timer.timeout.connect(ctrl.refresh)
    timer.start(500)

    ui.show()                       # paint the window FIRST, then load (MADIS pulls off-thread)

    def _deferred_init():
        ui.log("loading markets…")
        try:
            osm.seed_positions(api.get_positions())   # so a restart doesn't re-sell
        except Exception as e:
            ui.log(f"[init] position seed failed: {e}")
        ctrl.on_series_changed()    # fast REST; the slow MADIS pull is on the poll thread
    QTimer.singleShot(0, _deferred_init)

    rc = app.exec()

    strategy.stop()
    ws.stop()
    api.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
