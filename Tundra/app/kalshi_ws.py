"""Kalshi websocket feed — live orderbook BBO + own-account fills.

Trimmed port of Aston/feeds/ws_feed.py. Runs on a background daemon thread with
its own asyncio loop; subscribes orderbook_delta (+ fill) for the given tickers,
keeps a local book, and fires:
    on_update(ticker, yes_bid, yes_ask, bid_size, ask_size)   on every TOB change
    on_fill(ticker, action, side, price, count)               on every own fill
    on_stale()                                                 30s of no data (kill switch)
"""
import asyncio
import json
import threading
import time
from collections.abc import Callable

import websockets


class KalshiWsFeed:
    def __init__(self, api, on_update: Callable, on_fill: Callable | None = None,
                 on_stale: Callable | None = None):
        self.api = api
        self.on_update = on_update
        self.on_fill = on_fill
        self.on_stale = on_stale
        self.ws = None
        self.msg_id = 1
        self._loop = None
        self._running = False
        self.last_update_ts = 0.0
        self.books: dict[str, dict] = {}     # ticker -> {"yes_levels":{px:qty}, "no_levels":{px:qty}}
        self._tickers: list[str] = []

    def start(self, tickers: list[str]):
        self._tickers = list(tickers)
        self._running = True
        threading.Thread(target=self._run_loop, daemon=True, name="kalshi-ws").start()

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._schedule_close)

    def set_tickers(self, tickers: list[str]):
        """Replace the subscription set (called when the user changes selection)."""
        new = [t for t in tickers if t not in self.books]
        for t in new:
            self.books[t] = {"yes_levels": {}, "no_levels": {}}
            self._tickers.append(t)
        if new and self._running and self._loop:
            self._loop.call_soon_threadsafe(
                lambda t=new: asyncio.ensure_future(self._send_subscribe(t)))

    async def _send_subscribe(self, tickers):
        if not self.ws:
            return
        self.msg_id += 1
        try:
            await self.ws.send(json.dumps({
                "id": self.msg_id, "cmd": "subscribe",
                "params": {"channels": ["orderbook_delta"], "market_tickers": tickers}}))
        except Exception:
            pass

    def _schedule_close(self):
        if self.ws:
            asyncio.ensure_future(self._safe_close(), loop=self._loop)

    async def _safe_close(self):
        try:
            await self.ws.close()
        except Exception:
            pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        delay = 2
        while self._running:
            try:
                self._loop.run_until_complete(self._connect_and_listen())
            except RuntimeError:
                break
            except Exception as e:
                print(f"[WS] loop error: {e}")
            if not self._running:
                break
            print(f"[WS] disconnected, reconnecting in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)
        try:
            self._loop.close()
        except Exception:
            pass

    async def _connect_and_listen(self):
        try:
            self.ws = await websockets.connect(
                self.api.WS_URL, additional_headers=self.api.ws_auth_headers(),
                ping_interval=20, ping_timeout=10)
        except Exception as e:
            print(f"[WS] connect error: {e}")
            return
        channels = ["orderbook_delta"] + (["fill"] if self.on_fill else [])
        self.msg_id += 1
        await self.ws.send(json.dumps({
            "id": self.msg_id, "cmd": "subscribe",
            "params": {"channels": channels, "market_tickers": self._tickers}}))
        for t in self._tickers:
            self.books.setdefault(t, {"yes_levels": {}, "no_levels": {}})
        stale = asyncio.ensure_future(self._stale_monitor())
        try:
            async for message in self.ws:
                if not self._running:
                    break
                self._handle(json.loads(message))
        except (websockets.ConnectionClosed, Exception):
            pass
        finally:
            stale.cancel()
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _stale_monitor(self):
        while self._running:
            await asyncio.sleep(5)
            if self.last_update_ts and (time.time() - self.last_update_ts) > 30:
                print("[WS] STALE >30s -> reconnect")
                if self.on_stale:
                    try:
                        self.on_stale()
                    except Exception:
                        pass
                self.last_update_ts = 0.0
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                break

    def _handle(self, data: dict):
        self.last_update_ts = time.time()
        t = data.get("type")
        if t == "orderbook_snapshot":
            m = data["msg"]
            tk = m["market_ticker"]
            self.books[tk] = {
                "yes_levels": {float(p): float(q) for p, q in m.get("yes_dollars_fp", [])},
                "no_levels": {float(p): float(q) for p, q in m.get("no_dollars_fp", [])}}
            self._fire(tk)
        elif t == "orderbook_delta":
            m = data["msg"]
            tk = m["market_ticker"]
            if tk not in self.books:
                return
            key = "yes_levels" if m["side"] == "yes" else "no_levels"
            lv = self.books[tk][key]
            px, d = float(m["price_dollars"]), float(m["delta_fp"])
            q = lv.get(px, 0.0) + d
            if q <= 0:
                lv.pop(px, None)
            else:
                lv[px] = q
            self._fire(tk)
        elif t == "fill" and self.on_fill:
            m = data["msg"]
            price = 0.0
            for k in ("yes_price_dollars", "yes_price", "price"):
                if m.get(k):
                    price = float(m[k]); break
            try:
                self.on_fill(m.get("market_ticker", ""), m.get("action", ""),
                             m.get("side", ""), price,
                             int(float(m.get("count_fp", m.get("count", 0)))))
            except Exception as e:
                print(f"[WS] on_fill error: {e}")
        elif t == "error":
            print(f"[WS] server error: {data}")

    def _fire(self, ticker: str):
        b = self.books.get(ticker, {})
        ys, no = b.get("yes_levels", {}), b.get("no_levels", {})
        yes_bid = max(ys) if ys else 0.0
        bid_sz = int(ys.get(yes_bid, 0)) if yes_bid > 0 else 0
        if no:
            best_no = max(no)
            yes_ask = round(1.0 - best_no, 3)
            ask_sz = int(no.get(best_no, 0))
        else:
            yes_ask, ask_sz = 0.0, 0
        if yes_bid > 0 and yes_ask > 0 and yes_bid >= yes_ask:      # crossed = stale, resub
            self.books[ticker] = {"yes_levels": {}, "no_levels": {}}
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda t=ticker: asyncio.ensure_future(self._send_subscribe([t])))
            return
        try:
            self.on_update(ticker, yes_bid, yes_ask, bid_sz, ask_sz)
        except Exception:
            pass
