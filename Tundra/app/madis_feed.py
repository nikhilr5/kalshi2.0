"""MADIS real-time high-frequency (5-minute) ASOS feed.

MADIS LDAD/hfmetar = high-frequency METAR. The CURRENT-UTC-hour NetCDF file is
written live as 5-minute obs arrive, so polling it gives near-real-time 5-minute
temperatures -- the feed the floor strategy needs (IEM's 1-min archive lags ~2
days; the routine METAR is hourly). NetCDF *classic* format, parsed with scipy
(no netCDF4 dependency).

COVERAGE: most airport ASOS (verified: KMDW Chicago, KPHL Philadelphia, KEWR,
KORD, KLGA, ...). NOT covered: KNYC (Central Park) -- NYC's climate station does
not report hfmetar, so NYC's settlement station needs a different source.

RESOLUTION: ~5 minutes (not 1). The floor edge was backtested on 1-minute data;
re-validate at 5-min before relying on it (resample the 1-min archive to 5-min).

  feed = MadisFeed()
  df = feed.obs("KMDW")            # recent 5-min temps (ET)
  hi = feed.running_max_today("KMDW")
"""
import gzip
import io
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.io import netcdf_file

UA = {"User-Agent": "kalshi-weather (nikhil.richard84@gmail.com)"}
BASE = "https://madis-data.ncep.noaa.gov/madisPublic1/data/LDAD/hfmetar/netCDF"
# settlement stations that ARE in hfmetar (KNYC is not)
STATION_TZ = {"KMDW": "America/Chicago", "KPHL": "America/New_York",
              "KOKC": "America/Chicago", "KBOS": "America/New_York",
              "KDFW": "America/Chicago", "KDAL": "America/Chicago",
              "KLAX": "America/Los_Angeles", "KDEN": "America/Denver"}


def _decode_ids(arr):
    out = []
    for row in arr:
        b = b"".join(bytes([c]) if isinstance(c, (int, np.integer)) else bytes(c) for c in row)
        out.append(b.decode("ascii", "ignore").strip().strip("\x00").strip())
    return out


class MadisFeed:
    def __init__(self, session=None, tries=3):
        self.s = session or requests.Session()
        self.s.headers.update(UA)
        self.tries = tries
        self._cache = {}                         # hour-key -> parsed df (avoid re-pulling)

    def _hour_file(self, dt_utc):
        """Parse one hourly hfmetar file -> df[ts(UTC), station, tmpf, qc].
        Only COMPLETED hours are cached -- the current UTC hour's file is still
        being appended to as 5-min obs arrive, so caching it would freeze the feed
        (you'd re-serve a stale snapshot and never see new obs until the hour rolls)."""
        key = dt_utc.strftime("%Y%m%d_%H00")
        now_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H00")
        live = key == now_key
        if key in self._cache and not live:
            return self._cache[key]
        url = f"{BASE}/{key}.gz"
        raw = None
        for i in range(self.tries):
            try:
                r = self.s.get(url, timeout=60)
                if r.status_code == 200 and r.content[:2] == b"\x1f\x8b":   # gzip magic
                    raw = r.content
                    break
            except Exception:
                pass
            time.sleep(2 * (i + 1))
        if raw is None:
            df = pd.DataFrame(columns=["ts", "station", "tmpf", "qc"])
            if not live:
                self._cache[key] = df
            return df
        nc = netcdf_file(io.BytesIO(gzip.decompress(raw)), "r", mmap=False)
        ids = _decode_ids(nc.variables["stationId"][:])
        ot = nc.variables["observationTime"][:].astype("float64")
        tk = nc.variables["temperature"][:].astype("float64")
        qc = nc.variables["temperatureDD"][:] if "temperatureDD" in nc.variables else [b""] * len(ids)
        qc = [bytes([c]).decode("ascii", "ignore") if isinstance(c, (int, np.integer)) else
              (c.decode("ascii", "ignore") if isinstance(c, bytes) else str(c)) for c in qc]
        df = pd.DataFrame({
            "ts": pd.to_datetime(ot, unit="s", utc=True),
            "station": ids,
            "tmpf": (tk - 273.15) * 9 / 5 + 32,
            "qc": qc})
        df = df[(df["tmpf"] > -90) & (df["tmpf"] < 140)]      # drop fill/garbage
        if not live:
            self._cache[key] = df
        return df

    def obs(self, station, hours_back=2):
        """Recent 5-min obs for an ICAO station -> df[ts(local), tmpf, qc], sorted."""
        now = datetime.now(timezone.utc)
        frames = [self._hour_file(now - timedelta(hours=h)) for h in range(hours_back, -1, -1)]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if df.empty:
            return df
        df = df[df["station"] == station].copy()
        if df.empty:
            return df
        tz = STATION_TZ.get(station, "America/New_York")
        df["ts"] = df["ts"].dt.tz_convert(tz)
        return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    def running_max_today(self, station):
        """Max temp so far today (Local Standard Time day) -> (high_F, asof) or None.
        Pulls enough hours to cover the LST day. For a live bot, track incrementally
        instead of re-pulling all day every poll."""
        tz = STATION_TZ.get(station, "America/New_York")
        now_local = datetime.now(timezone.utc).astimezone()
        hrs = now_local.hour + 6                       # rough: cover since local midnight
        df = self.obs(station, hours_back=min(hrs, 26))
        if df.empty:
            return None
        today = df["ts"].iloc[-1].date()
        d = df[df["ts"].dt.date == today]
        if d.empty:
            return None
        return float(d["tmpf"].max()), d["ts"].iloc[-1].isoformat()


if __name__ == "__main__":
    feed = MadisFeed()

    while True:
        max = feed.running_max_today("KMDW")   # the number the floor strategy actually needs
        print(max)
        time.sleep(60)

