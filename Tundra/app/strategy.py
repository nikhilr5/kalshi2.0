"""Strategy — the weather-floor logic, owned by app.py.

Listens to the MADIS 5-min feed for the selected city/date, tracks the running
HIGH (max) or LOW (min) over the Local-Standard-Time climate day, and marks a
bucket DEAD the moment the extreme passes through it (high: run_max >= strike+1.5;
low: run_min <= strike-1.5). Dead buckets are provably worth $0 -> it tells OSM to
sell into the bid. Also holds the per-bucket state (live BBO from the websocket,
dead flag, position, expiry) that the UI renders.

All times handled/exposed in ET to match Kalshi. Cities are MADIS-covered only.
"""
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_BRE = re.compile(r"-B(\d+(?:\.\d+)?)$")
_NUMRE = re.compile(r"-?\d+(?:\.\d+)?")
_MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Safety cushion (°F) beyond the bracket edge before a bucket is called DEAD.
#   LOW  dead when run_min <= lo - cushion   (low has passed below the bracket)
#   HIGH dead when run_max >= hi + cushion   (high has passed above the bracket)
# MUST cover the FEED PRECISION GAP: MADIS reports whole °C (1°C = 1.8°F steps), but
# Kalshi settles on the official whole-°F NWS CLI. When MADIS reads 14°C (=57.2°F) the
# true/official low can be 57, 58 OR 59°F -- a ~2°F ambiguity WIDER than a bucket. So a
# bucket is only provably dead once MADIS is a full step (~2°F) past it; smaller cushions
# risk selling a WINNER (e.g. MADIS 57.2 'looks below 58' but the official low is 58 ->
# the 58-59 bucket WINS). 2.0 is the safe floor for this feed; do not go lower.
_DEFAULT_CUSHION = 2.0


def _bounds(sub_title: str, strike: float) -> tuple[float, float]:
    """(lo, hi) of the bracket from its sub_title (e.g. '61° to 62°'); falls back
    to the ticker midpoint +/-0.5 if the sub_title can't be parsed."""
    nums = [float(x) for x in _NUMRE.findall(sub_title or "")]
    if len(nums) >= 2:
        return min(nums[0], nums[1]), max(nums[0], nums[1])
    return strike - 0.5, strike + 0.5

# MADIS-covered cities with a validated floor edge. lst = Local STANDARD Time UTC
# offset (no DST) -> matches the NWS climate day Kalshi settles on.
CITIES: dict[str, dict] = {
    "Chicago":       dict(sta="KMDW", lst=-6, high="KXHIGHCHI",  low="KXLOWTCHI"),
    "Seattle":       dict(sta="KSEA", lst=-8, high="KXHIGHTSEA", low="KXLOWTSEA"),
    "Philadelphia":  dict(sta="KPHL", lst=-5, high="KXHIGHPHIL", low="KXLOWTPHIL"),
    "Miami":         dict(sta="KMIA", lst=-5, high="KXHIGHMIA",  low="KXLOWTMIA"),
    "Denver":        dict(sta="KDEN", lst=-7, high="KXHIGHDEN",  low="KXLOWTDEN"),
    "Oklahoma City": dict(sta="KOKC", lst=-6, high="KXHIGHTOKC", low="KXLOWTOKC"),
    "Boston":        dict(sta="KBOS", lst=-5, high="KXHIGHTBOS", low="KXLOWTBOS"),
    "Austin":        dict(sta="KAUS", lst=-6, high="KXHIGHAUS",  low="KXLOWTAUS"),
    "Dallas":        dict(sta="KDAL", lst=-6, high="KXHIGHTDAL", low="KXLOWTDAL"),
    "Los Angeles":   dict(sta="KLAX", lst=-8, high="KXHIGHLAX",  low="KXLOWTLAX"),
}


def _event_day_str(d) -> str:
    return f"{d.year % 100:02d}{_MON[d.month - 1]}{d.day:02d}"      # date -> 26JUN23


class Strategy:
    def __init__(self, api, osm, madis, log=print):
        self.api = api
        self.osm = osm
        self.madis = madis
        self.log = log
        self._lock = threading.Lock()
        self.city = None            # city key
        self.side = "low"           # "high" | "low"
        self.event_day = None       # "26JUN23"
        self.buckets: dict[str, dict] = {}      # ticker -> state dict
        self.extreme = None         # run_max if side=high else run_min
        self.run_max = None
        self.run_min = None
        self.asof_et = None
        self.n_obs = 0
        self.cushion = _DEFAULT_CUSHION     # °F past the bracket before DEAD (feed-precision guard)
        self._running = True
        self._wake = threading.Event()      # set -> poll thread refreshes now (off the UI thread)
        threading.Thread(target=self._poll_madis, daemon=True, name="madis-poll").start()

    # ---- selection ----
    def list_event_days(self, city: str, side: str) -> list[str]:
        """Open event-days (dates) for the city/side series, newest first."""
        series = CITIES[city][side]
        days = set()
        for m in self.api.get_markets(series_ticker=series, status="open"):
            tk = m.get("ticker", "")
            if "-B" in tk:
                parts = tk.split("-")
                if len(parts) >= 2:
                    days.add(parts[1])
        return sorted(days, key=self._day_sort_key, reverse=True)

    @staticmethod
    def _day_sort_key(ed: str):
        try:
            return datetime.strptime(ed, "%y%b%d")
        except ValueError:
            return datetime.min

    @staticmethod
    def today_event_day(city: str) -> str:
        """Today's climate-day event-day for the city (in its LST) — the date you
        actually trade, since that's where the running high/low is live."""
        lst = CITIES[city]["lst"]
        d = (datetime.now(timezone.utc) + timedelta(hours=lst)).date()
        return _event_day_str(d)

    def set_selection(self, city: str, side: str, event_day: str) -> list[str]:
        """Load the buckets for city/side/date. Returns the ticker list to subscribe."""
        series = CITIES[city][side]
        rows = {}
        for m in self.api.get_markets(series_ticker=series, status="open"):
            tk = m.get("ticker", "")
            mm = _BRE.search(tk)
            if not mm or event_day not in tk:
                continue
            close = m.get("close_time") or m.get("expiration_time")
            sub = m.get("yes_sub_title", "")
            lo, hi = _bounds(sub, float(mm.group(1)))
            rows[tk] = dict(
                ticker=tk, lo=lo, hi=hi, sub_title=sub,
                close_ts=close, yes_bid=0.0, yes_ask=0.0,
                bid_size=0, ask_size=0, dead=False)
        with self._lock:
            self.city, self.side, self.event_day = city, side, event_day
            self.buckets = rows
            self.extreme = self.run_max = self.run_min = None
            self.asof_et = None
            self.n_obs = 0
        self.log(f"[STRAT] {city} {side} {event_day}: {len(rows)} buckets")
        self._wake.set()       # poll thread pulls MADIS off the UI thread (first pull is slow)
        return list(rows.keys())

    # ---- MADIS running extreme ----
    def _lst_midnight_utc(self, ed: str, lst: int) -> datetime:
        d = datetime.strptime(ed, "%y%b%d")
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(hours=lst)

    def _refresh_extreme(self):
        with self._lock:
            city, ed = self.city, self.event_day
        if not city or not ed:
            return
        c = CITIES[city]
        start = self._lst_midnight_utc(ed, c["lst"])
        now = datetime.now(timezone.utc)
        hrs = max(2, min(int((now - start).total_seconds() // 3600) + 2, 30))
        try:
            df = self.madis.obs(c["sta"], hours_back=hrs)
        except Exception as e:
            self.log(f"[STRAT] madis error: {e}")
            return
        if df is None or df.empty:
            return
        u = df["ts"].dt.tz_convert("UTC")
        mask = (u >= start) & (u < start + timedelta(days=1))
        temps = df["tmpf"][mask]
        if temps.empty:
            return
        with self._lock:
            self.run_max = float(temps.max())
            self.run_min = float(temps.min())
            self.extreme = self.run_max if self.side == "high" else self.run_min
            self.asof_et = u[mask].max().tz_convert(_ET)
            self.n_obs = int(temps.shape[0])

    def _poll_madis(self):
        while self._running:
            try:
                self._refresh_extreme()
                self._evaluate()
            except Exception as e:
                self.log(f"[STRAT] poll error: {e}")
            self._wake.wait(60)        # wake immediately on a selection change, else every 60s
            self._wake.clear()

    # ---- live book ----
    def on_book(self, ticker, yes_bid, yes_ask, bid_size, ask_size):
        with self._lock:
            b = self.buckets.get(ticker)
            if not b:
                return
            b["yes_bid"], b["yes_ask"] = yes_bid, yes_ask
            b["bid_size"], b["ask_size"] = bid_size, ask_size
        self._evaluate_one(ticker)

    # ---- dead-bucket evaluation ----
    def _is_dead(self, b: dict) -> bool:
        """Dead = the running extreme has passed through the bracket (with cushion).
        LOW: run_min <= lo - CUSHION.  HIGH: run_max >= hi + CUSHION."""
        if self.extreme is None:
            return False
        if self.side == "high":
            return self.extreme >= b["hi"] + self.cushion
        return self.extreme <= b["lo"] - self.cushion

    def set_cushion(self, c: float):
        with self._lock:
            self.cushion = max(0.0, float(c))
        self._evaluate()

    def _evaluate_one(self, ticker: str):
        with self._lock:
            b = self.buckets.get(ticker)
            if not b or self.extreme is None:
                return
            b["dead"] = self._is_dead(b)
            dead, bid, bsz = b["dead"], b["yes_bid"], b["bid_size"]
        if dead and bid > 0:
            self.osm.want_short(ticker, bid, bsz)

    def _evaluate(self):
        with self._lock:
            tickers = list(self.buckets.keys())
        for tk in tickers:
            self._evaluate_one(tk)

    # ---- UI snapshot ----
    def snapshot(self) -> dict:
        now = datetime.now(timezone.utc)
        with self._lock:
            buckets = sorted(self.buckets.values(), key=lambda x: x["lo"])
            close_ts = buckets[0]["close_ts"] if buckets else None
            rows = [{"ticker": b["ticker"], "lo": b["lo"], "hi": b["hi"],
                     "sub_title": b["sub_title"], "yes_bid": b["yes_bid"],
                     "yes_ask": b["yes_ask"], "dead": b["dead"],
                     "pos": self.osm.position(b["ticker"])} for b in buckets]
            age = ((now - self.asof_et.to_pydatetime()).total_seconds()
                   if self.asof_et is not None else None)
            return dict(
                city=self.city, side=self.side, event_day=self.event_day,
                extreme=self.extreme, cushion=self.cushion,
                asof_et=self.asof_et.strftime("%H:%M:%S ET") if self.asof_et else "—",
                asof_age=age, n_obs=self.n_obs,
                tte=self._tte(close_ts, now),
                total_short=self.osm.total_short(), max_position=self.osm.max_position,
                buckets=rows)

    @staticmethod
    def _tte(close_ts, now) -> str:
        if not close_ts:
            return "—"
        try:
            ct = datetime.fromisoformat(close_ts.replace("Z", "+00:00"))
        except Exception:
            return "—"
        secs = int((ct - now).total_seconds())
        if secs <= 0:
            return "closed"
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def stop(self):
        self._running = False
