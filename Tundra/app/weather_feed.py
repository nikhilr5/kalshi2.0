"""Weather data client for the Tundra app.

One abstract source interface, one concrete NWS + IEM implementation. Each kind
of data the model needs is its own method so callers never touch a URL:

    forecast_highs(city)        deterministic multi-day daily-high forecast (NWS)
    current_high_today(city)    running max observed so far today  (IEM ASOS)
    hourly_temps(city, s, e)    historical/recent hourly temps     (IEM ASOS)
    mos_highs(city, model)      station-native MOS daily highs      (IEM MOS)
    observed_high(city, date)   official daily high for a past day  (IEM daily)

Swap in another source (NBM, ECMWF, a paid vendor) by subclassing WeatherSource
and implementing the same five methods; the model code stays unchanged.

Every network call is retry-once-then-skip and never raises on a bad
city/source -- callers get None / empty and a logged warning.
"""
import re
import time
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd

UA = {"User-Agent": "kalshi-weather (nikhil.richard84@gmail.com)"}

# city -> everything a source might need. station = settlement ICAO, asos = IEM id,
# network = IEM ASOS network, tz = IANA local zone, lst_off = Local STANDARD Time
# UTC offset (no DST) -- the NWS climate day Kalshi settles on is midnight-to-
# midnight LST, so the floor must bin obs by lst_off, not the DST-aware tz.
STATIONS = {
    "NYC": dict(series="KXHIGHNY",   lat=40.78, lon=-73.97, station="KNYC",
                asos="NYC", network="NY_ASOS", tz="America/New_York", lst_off=-5, name="Central Park, NY"),
    "LAX": dict(series="KXHIGHLAX",  lat=33.94, lon=-118.41, station="KLAX",
                asos="LAX", network="CA_ASOS", tz="America/Los_Angeles", lst_off=-8, name="LA Intl"),
    "OKC": dict(series="KXHIGHTOKC", lat=35.39, lon=-97.60, station="KOKC",
                asos="OKC", network="OK_ASOS", tz="America/Chicago", lst_off=-6, name="Will Rogers, OKC"),
    "BOS": dict(series="KXHIGHTBOS", lat=42.36, lon=-71.01, station="KBOS",
                asos="BOS", network="MA_ASOS", tz="America/New_York", lst_off=-5, name="Boston Logan"),
    "DAL": dict(series="KXHIGHTDAL", lat=32.85, lon=-96.85, station="KDAL",
                asos="DAL", network="TX_ASOS", tz="America/Chicago", lst_off=-6, name="Dallas Love Field"),
}


class WeatherSource(ABC):
    """Interface every data source implements. Methods are keyed by CITY code
    (see STATIONS); the source resolves the city to its own identifiers."""

    @abstractmethod
    def forecast_highs(self, city):
        """{date_iso: high_F} deterministic daily-high forecast, today forward."""

    @abstractmethod
    def current_high_today(self, city):
        """(high_so_far_F, n_obs, asof_local_iso) from today's observations, or None."""

    @abstractmethod
    def hourly_temps(self, city, start, end):
        """DataFrame[valid_utc, tmpf] of hourly temps over [start,end] (date strings)."""

    @abstractmethod
    def mos_highs(self, city, model="NBS"):
        """DataFrame[run_utc, target_local_date, lead_days, fc_high] station MOS highs."""

    @abstractmethod
    def observed_high(self, city, date):
        """Official observed daily high (F) for a past local date, or None."""


class NwsIemSource(WeatherSource):
    """Free stack: api.weather.gov for the forecast, IEM for obs / MOS / truth."""

    def __init__(self, session=None, polite_s=0.0):
        self.s = session or requests.Session()
        self.s.headers.update(UA)
        self.polite_s = polite_s

    # ---- transport ----------------------------------------------------------
    def _get(self, url, parse, tries=3, timeout=60):
        for i in range(tries):
            try:
                r = self.s.get(url, timeout=timeout)
                r.raise_for_status()
                out = parse(r)
                if self.polite_s:
                    time.sleep(self.polite_s)
                return out
            except Exception as e:
                if i + 1 >= tries:
                    print(f"[feed] GET failed {url[:80]}...: {e}")
                    return None
                time.sleep(2.0)
        return None

    @staticmethod
    def _tz(city):
        return ZoneInfo(STATIONS[city]["tz"])

    # ---- 1. forecast (NWS gridpoint) ---------------------------------------
    def forecast_highs(self, city):
        c = STATIONS.get(city)
        if not c:
            return None
        pts = self._get(f"https://api.weather.gov/points/{c['lat']},{c['lon']}",
                        lambda r: r.json())
        grid = (pts or {}).get("properties", {}).get("forecastGridData")
        if not grid:
            return None
        g = self._get(grid, lambda r: r.json())
        vals = (g or {}).get("properties", {}).get("maxTemperature", {}).get("values", [])
        out = {}
        for v in vals:
            cval, vt = v.get("value"), v.get("validTime", "")
            if cval is None or not vt:
                continue
            start = vt.split("/")[0]
            try:
                hh = int(start[11:13])
            except Exception:
                continue
            # keep the real daytime daily-high interval; drop overnight slivers
            # that carry the previous afternoon's heat (start 10-20 UTC, >=6h).
            if not (10 <= hh <= 20 and _iso_hours(vt) >= 6):
                continue
            day = start.split("T")[0]
            f = cval * 9 / 5 + 32
            out[day] = max(out.get(day, -999), f)
        return out or None

    # ---- 2. running high so far today (IEM ASOS) ---------------------------
    def current_high_today(self, city):
        """The LIVE floor: max observed temp so far today (real-time hourly METAR).
        Bins by Local STANDARD Time (lst_off, no DST) to match the NWS climate day
        Kalshi settles on -- an overnight reading just after DST-midnight belongs to
        the PREVIOUS standard-time day and must NOT count toward today's floor.
        Uses hourly METAR because it's the only near-real-time feed (1-min lags ~2d)."""
        c = STATIONS.get(city)
        if not c:
            return None
        z = timezone(timedelta(hours=c["lst_off"]))      # fixed standard offset, no DST
        # query yesterday..tomorrow (end is exclusive in IEM) so the whole LST
        # day is covered regardless of UTC offset, then filter to today LST
        now = datetime.now(z)
        df = self.hourly_temps(city, (now.date() - timedelta(days=1)).isoformat(),
                               (now.date() + timedelta(days=1)).isoformat())
        if df is None or df.empty:
            return None
        loc = pd.to_datetime(df["valid"], utc=True).dt.tz_convert(z)
        df = df[loc.dt.date == now.date()]
        if df.empty:
            return None
        hi = float(df["tmpf"].max())
        asof = pd.to_datetime(df["valid"], utc=True).dt.tz_convert(z).max()
        return hi, len(df), asof.isoformat()

    # ---- 2b. current observation (live poll: cloud cover, temp, etc.) ------
    def live_ob(self, city):
        """Current ob for a city from IEM /api/1/currents.json (the live-poll
        endpoint). Returns cloud cover (skyc1) + height (skyl1), temp, high-so-far
        (max_tmpf), present weather (wxcodes), the ob time, and the raw METAR."""
        c = STATIONS.get(city)
        if not c:
            return None
        js = self._get("https://mesonet.agron.iastate.edu/api/1/currents.json"
                       f"?network={c['network']}", lambda r: r.json())
        row = next((r for r in (js or {}).get("data", []) if r.get("station") == c["asos"]), None)
        if not row:
            return None
        keys = ["local_valid", "tmpf", "max_tmpf", "skyc1", "skyl1", "skyc2",
                "skyl2", "dwpf", "sknt", "drct", "wxcodes", "raw"]
        return {k: row.get(k) for k in keys}

    # ---- 3. hourly temps (IEM ASOS) ----------------------------------------
    def hourly_temps(self, city, start, end):
        c = STATIONS.get(city)
        if not c:
            return None
        sy, sm, sd = start.split("-")
        ey, em, ed = end.split("-")
        url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
               f"station={c['asos']}&data=tmpf&tz=Etc/UTC&format=onlycomma"
               f"&missing=empty&trace=empty"
               f"&year1={sy}&month1={int(sm)}&day1={int(sd)}"
               f"&year2={ey}&month2={int(em)}&day2={int(ed)}")
        df = self._get(url, lambda r: pd.read_csv(_io(r.text)), timeout=120)
        if df is None or "tmpf" not in df:
            return None
        df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
        return df.dropna(subset=["tmpf"])

    # ---- 4. station MOS daily highs (IEM) ----------------------------------
    def mos_highs(self, city, model="NBS"):
        c = STATIONS.get(city)
        if not c:
            return None
        url = (f"https://mesonet.agron.iastate.edu/api/1/mos.json?"
               f"station={c['station']}&model={model}")
        d = self._get(url, lambda r: r.json())
        rows = (d or {}).get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        rt = pd.to_datetime(df["runtime"], utc=True, errors="coerce")
        ft = pd.to_datetime(df["ftime"], utc=True, errors="coerce")
        z = self._tz(city)
        high = pd.to_numeric(df.get("tmp"), errors="coerce")
        for col in ("n_x", "txn"):
            if col in df:
                high = high.combine(pd.to_numeric(df[col], errors="coerce"),
                                    lambda a, b: max([x for x in (a, b) if pd.notna(x)] or [float("nan")]))
        m = pd.DataFrame({"rt": rt, "ft": ft, "high": high}).dropna(subset=["rt", "ft", "high"])
        m["target"] = m["ft"].dt.tz_convert(z).dt.date
        m["run_date"] = m["rt"].dt.tz_convert(z).dt.date
        g = m.groupby(["rt", "run_date", "target"], as_index=False)["high"].max()
        g["lead_days"] = (pd.to_datetime(g["target"]) - pd.to_datetime(g["run_date"])).dt.days
        return g.rename(columns={"rt": "run_utc", "target": "target_local_date",
                                 "high": "fc_high"})[
            ["run_utc", "target_local_date", "lead_days", "fc_high"]]

    # ---- 5. official observed daily high (CLI settlement value) ------------
    def observed_high(self, city, date):
        """Settlement-grade observed high (F) for a past local date. Prefers the
        NWS CLI value (what Kalshi settles on); falls back to the IEM ASOS daily
        max (~1F off CLI) if CLI is missing for that station/day."""
        cli = self.cli_high(city, date)
        if cli is not None:
            return cli
        c = STATIONS.get(city)
        if not c:
            return None
        y, mo, d = str(date).split("-")
        url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?"
               f"network={c['network']}&stations={c['asos']}"
               f"&year1={y}&month1={int(mo)}&day1={int(d)}"
               f"&year2={y}&month2={int(mo)}&day2={int(d)}"
               f"&var=max_temp_f&format=onlycomma")
        df = self._get(url, lambda r: pd.read_csv(_io(r.text)))
        if df is None or df.empty:
            return None
        v = pd.to_numeric(df["max_temp_f"], errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else None

    # ---- 6. CLI settlement value (NWS Climatological Report) ---------------
    def cli_high(self, city, date):
        """The official daily high (F) from the NWS CLI report -- the value
        Kalshi settles on. date='YYYY-MM-DD' local. None if no CLI for that
        station/day (e.g. KDAL has no Love Field CLI -> Dallas station unresolved).
        Published once per day (evening), so this is end-of-day truth, NOT a live
        intraday floor."""
        c = STATIONS.get(city)
        if not c:
            return None
        year = str(date)[:4]
        js = self._get(f"https://mesonet.agron.iastate.edu/json/cli.py?"
                       f"station={c['station']}&year={year}", lambda r: r.json())
        for row in (js or {}).get("results", []):
            if row.get("valid") == str(date):
                h = row.get("high")
                return float(h) if isinstance(h, (int, float)) else None
        return None

    # ---- 7. 1-minute ASOS max (historical precision) -----------------------
    def high_1min(self, city, date):
        """Max temp (F) from 1-MINUTE ASOS for a past local date -- catches the
        between-hour spikes the hourly METAR misses, so it's the most precise
        floor for BACKTESTING/calibration. NOTE: IEM's 1-min feed lags ~2 days,
        so it CANNOT serve the live floor (use current_high_today for that)."""
        c = STATIONS.get(city)
        if not c:
            return None
        url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?"
               f"station={c['asos']}&vars=tmpf&sts={date}T00:00Z&ets={date}T23:59Z"
               f"&sample=1min&what=download&tz=UTC&gis=no")
        df = self._get(url, lambda r: pd.read_csv(_io(r.text)), timeout=60)
        if df is None or df.empty or "tmpf" not in df:
            return None
        ts = pd.to_datetime(df["valid(UTC)"], utc=True, errors="coerce").dt.tz_convert(self._tz(city))
        v = pd.to_numeric(df["tmpf"], errors="coerce")[ts.dt.date.astype(str) == str(date)].dropna()
        return float(v.max()) if len(v) else None


def _iso_hours(valid_time):
    m = re.search(r"P(?:(\d+)D)?T?(?:(\d+)H)?", valid_time.split("/")[-1])
    return (int(m.group(1) or 0) * 24 + int(m.group(2) or 0)) if m else 0


def _io(text):
    import io
    return io.StringIO(text)


if __name__ == "__main__":
    src = NwsIemSource(polite_s=0.5)
    for city in ("LAX", "NYC"):
        print(f"\n=== {city} ({STATIONS[city]['station']}) ===")
        fh = src.forecast_highs(city)
        print("forecast highs:", {k: round(v) for k, v in sorted(fh.items())[:4]} if fh else None)
        print("current high today:", src.current_high_today(city))
        mos = src.mos_highs(city, "NBS")
        print("MOS rows:", None if mos is None else len(mos),
              "| latest lead0:", None if mos is None else
              mos[mos.lead_days == 0].tail(1).to_dict("records"))
        print("observed high 2026-06-15:", src.observed_high(city, "2026-06-15"))
