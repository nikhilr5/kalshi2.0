import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import zoneinfo

import numpy as np
import pandas as pd
import math
from statistics import NormalDist
import math
from scipy.optimize import brentq

SECONDS_PER_YEAR = 365.25 * 24 * 3600
ANN_MIN = 365.25 * 24 * 60
FOUR_LN2 = 4.0 * math.log(2.0)

# Where the recorder writes per-day DB files, and where this module
# caches anything pulled from S3.  Both overridable per-call.
DEFAULT_LOCAL_DIR = (Path("~/Desktop/Kalshi2.0/analysis/backtesting/data")
                     .expanduser())
DEFAULT_S3_CACHE_DIR = (Path("~/Desktop/Kalshi2.0/analysis/backtesting/_s3_cache")
                        .expanduser())
DEFAULT_S3_BUCKET = "s3://kalshibtc/archive"

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_day_suffix(suffix: str):
    """`26MAY15` → `date(2026, 5, 15)`.  Returns None on bad format."""
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})$", suffix)
    if not m:
        return None
    yy, mon, dd = m.groups()
    if mon not in _MONTHS:
        return None
    try:
        return date(2000 + int(yy), _MONTHS[mon], int(dd))
    except ValueError:
        return None


def list_eligible_dbs(series_prefix: str, cutoff: str,
                      local_dir: Path = DEFAULT_LOCAL_DIR,
                      s3_cache_dir: Path = DEFAULT_S3_CACHE_DIR,
                      s3_bucket: str = DEFAULT_S3_BUCKET) -> list[Path]:
    """Return every per-day DB path for `series_prefix` whose YYMONDD
    suffix is ≥ `cutoff`.  Local files take priority over S3.  Missing
    S3 files get downloaded into `s3_cache_dir` on demand."""
    cutoff_date = parse_day_suffix(cutoff)
    if cutoff_date is None:
        raise ValueError(f"bad cutoff format {cutoff!r}; expected YYMONDD")

    paths: dict[str, Path] = {}

    # Local — recorder's data dir
    for f in local_dir.glob(f"{series_prefix}-*.db"):
        suffix = f.stem.rsplit("-", 1)[-1]
        d = parse_day_suffix(suffix)
        if d and d >= cutoff_date:
            paths[f.name] = f

    # Local — previously-cached S3 files
    if s3_cache_dir.exists():
        for f in s3_cache_dir.glob(f"{series_prefix}-*.db"):
            if f.name in paths:
                continue
            suffix = f.stem.rsplit("-", 1)[-1]
            d = parse_day_suffix(suffix)
            if d and d >= cutoff_date:
                paths[f.name] = f

    # S3 — list and download anything not yet present locally
    try:
        r = subprocess.run(
            ["aws", "s3", "ls", f"{s3_bucket}/"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        s3_names = []
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[3]
            if (not name.startswith(f"{series_prefix}-")
                    or not name.endswith(".db")):
                continue
            suffix = name[:-3].rsplit("-", 1)[-1]
            d = parse_day_suffix(suffix)
            if d and d >= cutoff_date:
                s3_names.append(name)

        missing = [n for n in s3_names if n not in paths]
        if missing:
            s3_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[s3] downloading {len(missing)} file(s) → {s3_cache_dir}")
            for name in sorted(missing):
                target = s3_cache_dir / name
                print(f"     {name}")
                cp = subprocess.run(
                    ["aws", "s3", "cp", f"{s3_bucket}/{name}", str(target),
                     "--only-show-errors"],
                    capture_output=True, text=True,
                )
                if cp.returncode == 0:
                    paths[name] = target
                else:
                    print(f"     [warn] download failed: "
                          f"{cp.stderr.strip() or cp.stdout.strip()}")
    except FileNotFoundError:
        print("[s3] aws CLI not on PATH — using local files only")
    except subprocess.TimeoutExpired:
        print("[s3] aws s3 ls timed out — using local files only")
    except subprocess.CalledProcessError as e:
        print(f"[s3] aws s3 ls failed ({e.returncode}) — "
              f"using local files only")

    return sorted(paths.values(), key=lambda p: p.name)


def load_all_data(series_prefix: str, cutoff: str,
                  local_dir: Path = DEFAULT_LOCAL_DIR,
                  s3_cache_dir: Path = DEFAULT_S3_CACHE_DIR,
                  s3_bucket: str = DEFAULT_S3_BUCKET) -> tuple:
    """Concat theo / book / spot / fills / order_events across every
    eligible per-day DB.  Returns five DataFrames (in that order) with
    parsed `ts` columns.  Missing tables in older DBs are skipped."""
    files = list_eligible_dbs(series_prefix, cutoff,
                              local_dir, s3_cache_dir, s3_bucket)
    if not files:
        print(f"[load] no eligible DB files for {series_prefix} ≥ {cutoff}")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    print(f"[load] {len(files)} file(s) for {series_prefix} ≥ {cutoff}")
    theo_l, book_l, spot_l, fill_l, evt_l = [], [], [], [], []
    for path in files:
        print(f"   {path.name}")
        conn = sqlite3.connect(str(path))
        try:
            for table, target in (
                ("theo_state",   theo_l),
                ("kalshi_book",  book_l),
                ("spot_ticks",   spot_l),
                ("fills",        fill_l),
                ("order_events", evt_l),
            ):
                try:
                    target.append(
                        pd.read_sql(f"SELECT * FROM {table} ORDER BY ts", conn))
                except Exception as e:
                    print(f"     [warn] skipping {table}: {e}")
        finally:
            conn.close()

    def _concat(parts):
        return (pd.concat(parts, ignore_index=True)
                if parts else pd.DataFrame())

    theo  = _concat(theo_l)
    book  = _concat(book_l)
    spot  = _concat(spot_l)
    fills = _concat(fill_l)
    events = _concat(evt_l)

    for df in (theo, book, spot, fills, events):
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return theo, book, spot, fills, events


def realized_sigma_forward(spot: pd.DataFrame,
                            horizon_minutes: int = 15) -> pd.DataFrame:
    """Per-minute forward-looking Parkinson σ from spot ticks.

    Returns a DataFrame with columns:
        minute            — 1-min boundary (UTC)
        high, low         — within-minute extremes
        parkinson_var     — per-minute variance: ln(H/L)² / (4·ln 2)
        realized_{N}m     — σ over the NEXT `horizon_minutes` minutes,
                            annualized (NaN for trailing rows that
                            don't have a full forward window).
    """
    if spot.empty:
        return pd.DataFrame()
    spot = spot.sort_values('ts').reset_index(drop=True)
    spot = spot.assign(minute=spot['ts'].dt.floor('1min'))
    minute_bars = (
        spot.groupby('minute')
            .agg(high=('price', 'max'), low=('price', 'min'))
            .reset_index()
            .sort_values('minute')
            .reset_index(drop=True)
    )
    hl_ratio = minute_bars['high'] / minute_bars['low']
    minute_bars['parkinson_var'] = np.where(
        (minute_bars['high'] > 0)
        & (minute_bars['low'] > 0)
        & (minute_bars['high'] > minute_bars['low']),
        np.log(hl_ratio) ** 2 / FOUR_LN2,
        0.0,
    )
    cum = minute_bars['parkinson_var'].cumsum()
    future_var = cum.shift(-horizon_minutes) - cum
    col = f'realized_{horizon_minutes}m'
    minute_bars[col] = np.sqrt(
        future_var.clip(lower=0) * (ANN_MIN / horizon_minutes)
    )
    minute_bars.loc[future_var.isna(), col] = np.nan
    return minute_bars


def fetch_settlements_from_api(tickers, kalshi_api,
                                 cache_path: Path | str | None = None
                                 ) -> dict[str, int]:
    """Pull authoritative settlement results from Kalshi.

    Returns `{ticker: 1 if result == "yes" else 0}` for every ticker
    whose market has settled.  Tickers without a `result` field yet
    (not closed, or closed-but-not-yet-resolved) are silently omitted.

    Caches results to `cache_path` as JSON.  Settled markets are
    immutable, so subsequent runs re-use the cache and only hit the
    API for tickers that haven't been seen.  Pass `cache_path=None`
    to disable caching entirely (always queries the API).
    """
    cache: dict[str, int] = {}
    cache_p: Path | None = Path(cache_path) if cache_path else None
    if cache_p and cache_p.exists():
        try:
            with cache_p.open() as f:
                cache = {k: int(v) for k, v in json.load(f).items()}
        except Exception as e:
            print(f"[settle] cache read failed: {e}")
            cache = {}

    out: dict[str, int] = {}
    new_lookups = 0
    new_settled = 0
    for ticker in tickers:
        if ticker in cache:
            out[ticker] = cache[ticker]
            continue
        new_lookups += 1
        try:
            market = kalshi_api.get_market(ticker)
        except Exception as e:
            print(f"[settle] {ticker}: API error: {e}")
            continue
        result = (market.get("result") or "").lower()
        if result == "yes":
            out[ticker] = cache[ticker] = 1
            new_settled += 1
        elif result == "no":
            out[ticker] = cache[ticker] = 0
            new_settled += 1
        # else: not yet settled — leave out

    if new_settled > 0 and cache_p is not None:
        try:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            with cache_p.open("w") as f:
                json.dump(cache, f)
        except Exception as e:
            print(f"[settle] cache write failed: {e}")

    print(f"[settle] {len(out)} settled / {len(tickers)} tickers "
          f"({new_lookups} new lookups, {new_settled} newly settled)")
    return out


def compute_settlements(theo: pd.DataFrame, spot: pd.DataFrame,
                          twap_window_s: int = 60,
                          min_ticks: int = 5) -> dict[str, int]:
    """Approximate per-ticker settlement outcomes from spot ticks.

    Kalshi crypto 15-min markets settle on the TWAP of the underlying
    over the final `twap_window_s` seconds before close.  We don't
    capture the official settlement yet, so we derive it: average
    `spot_ticks.price` over [close − twap_window_s, close], compare to
    strike, emit 1 if TWAP > strike else 0.

    Tickers with fewer than `min_ticks` spot ticks in the window are
    skipped (returned dict doesn't include them) — they'd be too noisy
    to score reliably.
    """
    if theo.empty or spot.empty:
        return {}
    spot = spot.sort_values('ts')
    out: dict[str, int] = {}
    for ticker, group in theo.groupby('ticker'):
        try:
            last = group.loc[group['seconds_to_expiry'].idxmin()]
        except (KeyError, ValueError):
            continue
        secs_remaining = float(last['seconds_to_expiry'])
        close_time = last['ts'] + pd.Timedelta(seconds=secs_remaining)
        strike = float(last['strike'])
        if strike <= 0:
            continue
        window_start = close_time - pd.Timedelta(seconds=twap_window_s)
        twap_rows = spot[(spot['ts'] >= window_start)
                         & (spot['ts'] <= close_time)]
        if len(twap_rows) < min_ticks:
            continue
        twap = float(twap_rows['price'].mean())
        out[ticker] = 1 if twap > strike else 0
    return out


def brier_score(predictions: pd.Series, outcomes: pd.Series) -> float | None:
    """Brier = mean((pred − outcome)²).  Returns None if no valid pairs.

    Lower is better.  A naive constant-0.5 forecaster scores 0.25;
    perfect calibration scores 0.  Used to compare probability
    forecasts (theo, market mid) against binary settlement outcomes.
    """
    mask = (~predictions.isna()) & (~outcomes.isna())
    p, o = predictions[mask], outcomes[mask]
    if len(p) == 0:
        return None
    return float(((p - o) ** 2).mean())


def forecast_error_stats(pred: pd.Series, actual: pd.Series,
                          sep: str = "\n") -> str:
    """Multi-line formatted summary of forecast accuracy.

    Reports n, correlation, bias, MAE, RMSE — same definitions used in
    vol_forecasting's calibration panels.  `sep="<br>"` for HTML / plotly
    annotations, `sep="\\n"` (default) for terminal printing."""
    mask = (~pred.isna()) & (~actual.isna())
    p, a = pred[mask], actual[mask]
    if len(p) == 0:
        return "n=0"
    err = p - a
    parts = [
        f"n={len(p):,}",
        f"corr ={np.corrcoef(p, a)[0,1]:+.3f}",
        f"bias ={err.mean()*100:+.2f}%",
        f"MAE  ={err.abs().mean()*100:.2f}%",
        f"RMSE ={np.sqrt((err**2).mean())*100:.2f}%",
    ]
    return sep.join(parts)


def implied_sigma(market_price, spot, strike, seconds_to_expiry, r: float = 0.0):
    """Closed-form annualized σ implied by N(d2) = market_price.

    Works on scalars, numpy arrays, or pandas Series — all broadcast.
    Returns same type, with NaN for unsolvable inputs (price beyond
    the model's max for the given moneyness/T, or degenerate inputs).

    Derivation: let u = σ√T.  d2 = m/u − u/2 where m = ln(S/K) + rT.
    Setting d2 = N⁻¹(price) = x:
        u² + 2xu − 2m = 0   →   u = −x + √(x² + 2m)
    σ = u / √T.  When the discriminant is negative, the binary's
    price is outside the no-drift lognormal's reachable range — return
    NaN.

    Much faster than brentq on a per-row basis (~50× on 1M rows).
    """
    import numpy as np
    from scipy.stats import norm as _norm

    price = np.asarray(market_price, dtype=float)
    S     = np.asarray(spot,         dtype=float)
    K     = np.asarray(strike,       dtype=float)
    secs  = np.asarray(seconds_to_expiry, dtype=float)

    eps = 1e-6
    valid = ((price > eps) & (price < 1 - eps)
             & (S > 0) & (K > 0) & (secs > 0))

    # Safe inputs so the math doesn't warn on invalid rows; we'll
    # mask the output with NaN at the end.
    safe_price = np.where(valid, price, 0.5)
    safe_S     = np.where(valid, S, 1.0)
    safe_K     = np.where(valid, K, 1.0)
    safe_T     = np.where(valid, secs / SECONDS_PER_YEAR, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        x = _norm.ppf(safe_price)
        m = np.log(safe_S / safe_K) + r * safe_T
        disc = x * x + 2 * m
        sqrt_disc = np.sqrt(np.where(disc > 0, disc, 0))
        # Two positive roots possible.  The larger σ root is the
        # pathological one (drift correction dominates); the lower
        # is the conventional IV.  Pick the smaller positive u.
        u_plus  = -x + sqrt_disc
        u_minus = -x - sqrt_disc
        u = np.where(u_minus > 0, u_minus, u_plus)
        sigma = u / np.sqrt(safe_T)

    out = np.where(valid & (disc > 0) & (u > 0), sigma, np.nan)
    # Preserve pandas Series indexing if the caller passed one.
    if hasattr(market_price, "index"):
        import pandas as pd
        return pd.Series(out, index=market_price.index)
    return out

#used to compute markouts when passed in fills and book
def calculate_markouts(fills : pd.DataFrame, book: pd.DataFrame, markouts: list[int]):
    result = fills
    for t in markouts:
        new_field = 'markout_' + str(t) + 's'
        result[new_field + "_ts"] = result['ts'] + pd.Timedelta(seconds=t)
        result = pd.merge_asof(
            result,
            book,
            left_on=new_field + '_ts',
            right_on='ts',
            by='ticker',
            direction='backward',
            suffixes=('_f', '_' + new_field)
        )
        result['yes_mid'] = (result['yes_bid'] + result['yes_ask']) / 2
        result[new_field] = np.where(result['action'] == 'buy',
            (result['yes_mid']- result['price'])* 100,
            (result['price'] - result['yes_mid']) * 100)
        
        result['ts'] = result['ts_f']
        result = result.drop(columns=['yes_bid', 'yes_ask', 'yes_mid', 'ts_f', 'id', 'ask_size', 'bid_size'])
        
    return result

@dataclass(frozen=True)
class AnalysisUtils:
    """
    Small utilities for pulling/transforming recorder DB data for analysis notebooks/scripts.
    """

    db_path: str = "../marketdata/recorder.db"
    tz_local: str = "America/Chicago"

    def _to_utc_iso(self, dt: datetime) -> str:
        tz = zoneinfo.ZoneInfo(self.tz_local)
        dt = dt if dt.tzinfo else dt.replace(tzinfo=tz)
        return dt.astimezone(timezone.utc).isoformat()

    def load_market_snapshots(
        self,
        start: datetime,
        end: datetime,
        *,
        event_prefix: str = "KXBTCD",
    ) -> pd.DataFrame:
        """
        Return all market_snapshots rows for every ticker in the time window.
        """
        start_utc = self._to_utc_iso(start)
        end_utc = self._to_utc_iso(end)

        sql = """
        SELECT *
        FROM market_snapshots
        WHERE ts >= ? AND ts < ?
          AND event_ticker LIKE ?
        ORDER BY ts
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(sql, conn, params=(start_utc, end_utc, f"{event_prefix}%"))

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values(["event_ticker", "ticker", "ts"])
        return df

    def load_one_market_snapshot(
        self,
        start: datetime,
        end: datetime,
        *,
        event_prefix: str = "KXBTCD",
    ) -> pd.DataFrame:
        """
        Return market_snapshots for a single ticker per day, covering the full range.

        Picks the ticker with the most data each day so spot_mid has no gaps.
        Much lighter than load_market_snapshots for spot-only analysis.
        """
        start_utc = self._to_utc_iso(start)
        end_utc = self._to_utc_iso(end)

        with sqlite3.connect(self.db_path) as conn:
            day_tickers = conn.execute("""
                SELECT substr(ts, 1, 10) AS day, ticker, COUNT(*) AS n
                FROM market_snapshots
                WHERE ts >= ? AND ts < ? AND event_ticker LIKE ?
                GROUP BY day, ticker
                ORDER BY day, n DESC
            """, (start_utc, end_utc, f"{event_prefix}%")).fetchall()

            if not day_tickers:
                return pd.DataFrame()

            best_per_day = {}
            for day, ticker, n in day_tickers:
                if day not in best_per_day:
                    best_per_day[day] = ticker

            tickers_needed = list(set(best_per_day.values()))
            placeholders = ",".join("?" * len(tickers_needed))

            sql = f"""
            SELECT ts, event_ticker, ticker, strike, spot_mid, spot_bid, spot_ask,
                   kalshi_yes_bid, kalshi_yes_ask, theo_bid, theo_ask,
                   deribit_bid_iv, deribit_ask_iv, close_time
            FROM market_snapshots
            WHERE ts >= ? AND ts < ?
              AND ticker IN ({placeholders})
            ORDER BY ts
            """
            df = pd.read_sql(sql, conn, params=[start_utc, end_utc] + tickers_needed)

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values(["event_ticker", "ticker", "ts"])
        return df

    @staticmethod
    def snap_to_grid(df: pd.DataFrame, *, freq: str = "5min", tolerance: str = "2min") -> pd.DataFrame:
        """Snap each (event_ticker, ticker) series to nearest fixed time grid."""
        if df.empty:
            return df.copy()

        tol = pd.Timedelta(tolerance)
        parts = []
        for (event_ticker, ticker), g in df.groupby(["event_ticker", "ticker"], sort=False):
            g = g.sort_values("ts")
            grid = pd.DataFrame(
                {
                    "grid_ts": pd.date_range(
                        g["ts"].min().floor(freq),
                        g["ts"].max().ceil(freq),
                        freq=freq,
                        tz="UTC",
                    )
                }
            )
            snapped = pd.merge_asof(
                grid.sort_values("grid_ts"),
                g.sort_values("ts"),
                left_on="grid_ts",
                right_on="ts",
                direction="nearest",
                tolerance=tol,
            )
            snapped["event_ticker"] = event_ticker
            snapped["ticker"] = ticker
            parts.append(snapped)

        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        out = out.dropna(subset=["ts"]) if not out.empty else out
        return out.sort_values(["event_ticker", "ticker", "grid_ts"])

    @staticmethod
    def implied_vol_binary(price: float, S: float, K: float, T: float,
                        r: float = 0.0) -> float:
        """Closed-form quadratic IV for a binary above option.

        Given P(above K) = N(d2), invert for sigma:
            x = N_inv(P),  m = ln(S/K) + rT
            u^2 + 2xu - 2m = 0  ->  u = -x + sqrt(x^2 + 2m)
            sigma = u / sqrt(T)

        Args:
            price: observed binary option price (0-1)
            S: spot price
            K: strike price
            T: time to expiry in years
            r: risk-free rate

        Returns IV as decimal (e.g. 0.65 for 65%). Returns 0.0 if unsolvable.
        """

        if price <= 0.01 or price >= 0.99 or S <= 0 or K <= 0 or T <= 0:
            return 0.0
        try:
            norm = NormalDist()
            x = norm.inv_cdf(price)
            m = math.log(S / K) + r * T
            disc = x * x + 2 * m
            if disc < 0:
                return 0.0
            sqrt_disc = math.sqrt(disc)
            u1 = -x + sqrt_disc
            u2 = -x - sqrt_disc
            candidates = [u for u in (u1, u2) if u > 0]
            if not candidates:
                return 0.0
            u = min(candidates)
            return u / math.sqrt(T)
        except Exception:
            return 0.0

    @staticmethod
    def fit_vol_smile(strikes: np.ndarray, ivs: np.ndarray) -> tuple[float, float, float]:
        """Fit a parabola (degree-2 poly) to IV vs strike.

        Args:
            strikes: strike prices (x-axis)
            ivs: implied vols corresponding to each strike

        Returns (a, b, c) where IV ≈ a*K² + b*K + c.
        """
        mask = ivs > 0
        coeffs = np.polyfit(strikes[mask], ivs[mask], 2)
        return coeffs[0], coeffs[1], coeffs[2]

    @staticmethod
    def plot_vol_smile(a: float, b: float, c: float,
                       strikes: np.ndarray, ivs: np.ndarray | None = None,
                       fitted_ivs: np.ndarray | None = None,
                       title: str = "Vol Smile") -> None:
        """Plot fitted parabola and optionally overlay raw and fitted IV points.

        Args:
            a, b, c: coefficients from fit_vol_smile
            strikes: strike array to define x-axis range
            ivs: if provided, scatter the raw IVs on the same plot
            fitted_ivs: if provided, scatter the fitted IVs at each strike
            title: chart title
        """
        import matplotlib.pyplot as plt

        x = np.linspace(strikes.min(), strikes.max(), 200)
        y = a * x**2 + b * x + c

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, "r-", linewidth=2, label="Fitted parabola")
        if ivs is not None:
            plt.scatter(strikes, ivs, s=30, zorder=5, label="Observed IV")
        if fitted_ivs is not None:
            plt.scatter(strikes, fitted_ivs, s=40, marker="x", zorder=6, label="Fitted IV")
        plt.xlabel("Strike")
        plt.ylabel("Implied Vol")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def graph_strike(df: pd.DataFrame, strike: float = None, iv_cols: list[str] | None = None,
                     show_spot: bool = True, use_real_trades: bool = False) -> None:
        """Launch interactive Dash app with buy/sell edge inputs for a single strike.

        If use_real_trades=True, fetches historical trades from Kalshi API and
        uses those for markout analysis instead of theoretical signals.
        """
        from dash import Dash, dcc, html, Input, Output, State, Patch
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        if iv_cols is None:
            iv_cols = ["smoothed_mid_iv"]

        # Pre-fetch real trades per ticker if requested
        _trade_cache = {}
        if use_real_trades:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "4RunnerApp2.0"))
            from kalshi_api import KalshiAPI
            _api = KalshiAPI()
            for ticker in df["ticker"].unique():
                try:
                    trades = _api.get_trades(ticker, limit=5000)
                    _trade_cache[ticker] = trades
                    print(f"[Trades] {ticker}: {len(trades)} trades")
                except Exception as e:
                    print(f"[Trades] {ticker}: failed ({e})")
                    _trade_cache[ticker] = []

        # Extract event tickers from the ticker column
        if "ticker" in df.columns:
            available_events = sorted(df["ticker"].str.extract(r'(KXBTCD-[^-]+)')[0].dropna().unique())
        else:
            available_events = ["ALL"]
        default_event = available_events[0] if available_events else "ALL"

        available_strikes = sorted(df["strike"].unique())
        if strike is None:
            strike = available_strikes[len(available_strikes) // 2]

        # detect available spans from columns
        available_spans = sorted([int(c.split("_")[-1]) for c in df.columns if c.startswith("theo_fitted_")])
        if not available_spans:
            available_spans = [60]

        app = Dash(__name__)
        app.layout = html.Div([
            html.Div([
                html.Label("Event: "),
                dcc.Dropdown(id="event", options=[{"label": e, "value": e} for e in available_events],
                             value=default_event, clearable=False, style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"}),
                html.Label(" Strike: ", style={"marginLeft": "20px"}),
                dcc.Dropdown(id="strike", options=[{"label": f"{k:.0f}", "value": k} for k in available_strikes],
                             value=strike, clearable=False, style={"width": "140px", "display": "inline-block", "verticalAlign": "middle"}),
                html.Label(" Buy Edge: ", style={"marginLeft": "20px"}), dcc.Input(id="buy-edge", type="number", value=0, step=0.01, style={"width": "80px"}),
                html.Label(" Sell Edge: ", style={"marginLeft": "20px"}), dcc.Input(id="sell-edge", type="number", value=0, step=0.01, style={"width": "80px"}),
                html.Label(" Smoothing Span: ", style={"marginLeft": "20px"}),
                dcc.Dropdown(id="span", options=[{"label": str(sp), "value": sp} for sp in available_spans],
                             value=60, clearable=False, style={"width": "100px", "display": "inline-block", "verticalAlign": "middle"}),
                html.Label(" IV Min: ", style={"marginLeft": "20px"}), dcc.Input(id="iv-min", type="number", value=0, step=1, style={"width": "60px"}),
                html.Label(" IV Max: ", style={"marginLeft": "10px"}), dcc.Input(id="iv-max", type="number", value=100, step=1, style={"width": "60px"}),
            ], style={"padding": "10px", "display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
            dcc.Graph(id="strike-graph", style={"height": "90vh"}),
            html.Div(id="markout-stats",
                     style={"padding": "10px", "fontFamily": "monospace",
                            "fontSize": "14px", "color": "#333"}),
            dcc.Store(id="markout-store"),
        ])

        # Update strike dropdown when event changes
        @app.callback(
            Output("strike", "options"),
            Output("strike", "value"),
            Input("event", "value"),
        )
        def update_strikes_for_event(selected_event):
            if selected_event and selected_event != "ALL":
                ev_df = df[df["ticker"].str.contains(selected_event, na=False)]
            else:
                ev_df = df
            ev_strikes = sorted(ev_df["strike"].unique())
            options = [{"label": f"{k:.0f}", "value": k} for k in ev_strikes]
            # Default to nearest ATM
            spot = ev_df["spot_mid"].median() if not ev_df.empty else 0
            if spot > 0 and ev_strikes:
                default = min(ev_strikes, key=lambda s: abs(s - spot))
            elif ev_strikes:
                default = ev_strikes[len(ev_strikes) // 2]
            else:
                default = None
            return options, default

        @app.callback(Output("strike-graph", "figure"),
                       Output("markout-store", "data"),
                       Input("event", "value"), Input("strike", "value"),
                       Input("buy-edge", "value"),
                       Input("sell-edge", "value"), Input("span", "value"),
                       Input("iv-min", "value"), Input("iv-max", "value"))
        def update_graph(selected_event, selected_strike, buy_edge, sell_edge, span, iv_min, iv_max):
            buy_edge = buy_edge or 0
            sell_edge = sell_edge or 0
            span = span or 60
            selected_strike = selected_strike or strike
            theo_col = f"theo_fitted_{span}"
            iv_col = f"smoothed_mid_iv_{span}"

            # Filter by event
            if selected_event and selected_event != "ALL":
                ev_df = df[df["ticker"].str.contains(selected_event, na=False)]
            else:
                ev_df = df
            s = ev_df[ev_df["strike"] == selected_strike].sort_values("ts").reset_index(drop=True)

            markout_labels = ["1m Markout", "5m Markout", "10m Markout", "30m Markout"]
            specs = [[{"secondary_y": True}], [{"secondary_y": True}]] + [[{}]] * 4 + [[{}]]
            fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                row_heights=[3, 1.5, 1, 1, 1, 1, 1.5],
                                subplot_titles=[f"Strike {selected_strike:.0f} — Market vs Fitted Theo",
                                                f"Smoothed IV over Time"] + markout_labels + ["Cumulative PnL"],
                                specs=specs)

            if s.empty:
                return fig, {}

            fig.add_trace(go.Scatter(x=s["ts"], y=s["kalshi_yes_bid"], name="Kalshi Bid",
                                     line=dict(color="blue"), opacity=0.6), row=1, col=1)
            fig.add_trace(go.Scatter(x=s["ts"], y=s["kalshi_yes_ask"], name="Kalshi Ask",
                                     line=dict(color="red"), opacity=0.6), row=1, col=1)

            theo = s[theo_col] if theo_col in s.columns else s["theo_fitted"]
            iv_vals = s[iv_col] if iv_col in s.columns else s.get("smoothed_mid_iv", pd.Series([float('nan')] * len(s)))

            hover_text = [f"Theo: {t:.3f}<br>Smoothed IV: {iv:.1%}" if iv == iv else f"Theo: {t:.3f}"
                          for t, iv in zip(theo, iv_vals)]
            fig.add_trace(go.Scatter(x=s["ts"], y=theo, name=f"Fitted Theo (span={span})",
                                     line=dict(color="black", width=2),
                                     hovertext=hover_text, hoverinfo="text+x"), row=1, col=1)

            if buy_edge > 0 or sell_edge > 0:
                buy_theo = theo - buy_edge
                sell_theo = theo + sell_edge

                if buy_edge > 0:
                    fig.add_trace(go.Scatter(x=s["ts"], y=buy_theo, name="Buy Theo",
                                             line=dict(color="black", width=1, dash="dot"), opacity=0.4), row=1, col=1)

                if sell_edge > 0:
                    fig.add_trace(go.Scatter(x=s["ts"], y=sell_theo, name="Sell Theo",
                                             line=dict(color="black", width=1, dash="dot"), opacity=0.4), row=1, col=1)

            if show_spot:
                fig.add_trace(go.Scatter(x=s["ts"], y=s["spot_mid"], name="BTC Spot",
                                         line=dict(color="green"), opacity=0.6),
                              row=1, col=1, secondary_y=True)

            if iv_col in s.columns:
                fig.add_trace(go.Scatter(x=s["ts"], y=s[iv_col] * 100, name=f"Smoothed IV (span={span})"), row=2, col=1)

            # --- Markout rows (3-6) ---
            markout_intervals = [(60, "1m"), (300, "5m"), (600, "10m"), (1800, "30m")]

            # Collect fills: real trades or theoretical signals
            theo_fills = []

            if use_real_trades and _trade_cache:
                # Use real historical trades, filter to ones where we'd have edge
                ticker = s["ticker"].iloc[0] if "ticker" in s.columns else None
                real_trades = _trade_cache.get(ticker, []) if ticker else []

                # Plot ALL real trades on row 1 (light color, before filtering)
                # Filter to snapshot time range
                snap_ts_all = pd.to_datetime(s["ts"], utc=True).dt.tz_localize(None).values
                _snap_min = snap_ts_all[0]
                _snap_max = snap_ts_all[-1]
                if real_trades:
                    all_ts = []
                    all_px = []
                    all_colors = []
                    for t in real_trades:
                        px = float(t.get("yes_price_dollars", 0))
                        if px <= 0:
                            continue
                        ts_val = pd.Timestamp(t["created_time"])
                        if ts_val.tzinfo:
                            ts_val = ts_val.tz_localize(None)
                        if ts_val.to_datetime64() < _snap_min or ts_val.to_datetime64() > _snap_max:
                            continue
                        all_ts.append(ts_val)
                        all_px.append(px)
                        # cyan for yes taker, orange for no taker
                        all_colors.append("#06b6d4" if t.get("taker_side") == "yes" else "#f97316")
                    if all_ts:
                        fig.add_trace(go.Scatter(
                            x=all_ts, y=all_px,
                            mode="markers", name="All Trades",
                            marker=dict(size=4, color=all_colors, opacity=0.4),
                        ), row=1, col=1)

                # Build theo lookup: snap timestamps → theo values
                snap_ts_list = pd.to_datetime(s["ts"], utc=True).dt.tz_localize(None).values
                theo_vals = theo.values
                snap_min = snap_ts_list[0]
                snap_max = snap_ts_list[-1]

                from bisect import bisect_left as _bl

                for t in real_trades:
                    trade_ts = pd.Timestamp(t["created_time"])
                    if trade_ts.tzinfo:
                        trade_ts = trade_ts.tz_localize(None)
                    trade_price = float(t.get("yes_price_dollars", 0))
                    taker_side = t.get("taker_side", "")
                    if trade_price <= 0:
                        continue

                    # Skip trades outside snapshot time range
                    trade_ts_np = trade_ts.to_datetime64()
                    if trade_ts_np < snap_min or trade_ts_np > snap_max:
                        continue

                    # Find nearest snapshot to get theo at trade time
                    idx = _bl(snap_ts_list, trade_ts_np)
                    idx = min(idx, len(snap_ts_list) - 1)
                    theo_at_trade = theo_vals[idx]

                    # Determine if we'd be the maker on this trade
                    # taker_side="yes" means someone bought yes → we'd be the sell side (maker)
                    # taker_side="no" means someone sold yes → we'd be the buy side (maker)
                    if taker_side == "yes" and sell_edge > 0:
                        # Someone bought from us — check if trade price >= our sell theo
                        if trade_price >= theo_at_trade + sell_edge:
                            theo_fills.append(("sell", trade_ts, trade_price))
                    elif taker_side == "no" and buy_edge > 0:
                        # Someone sold to us — check if trade price <= our buy theo
                        if trade_price <= theo_at_trade - buy_edge:
                            theo_fills.append(("buy", trade_ts, trade_price))
            else:
                # Theoretical signals from snapshot data
                if buy_edge > 0:
                    buy_theo = theo - buy_edge
                    buy_signals = s[s["kalshi_yes_ask"] <= buy_theo]
                    for _, row_data in buy_signals.iterrows():
                        theo_fills.append(("buy", row_data["ts"], row_data["kalshi_yes_ask"]))
                if sell_edge > 0:
                    sell_theo = theo + sell_edge
                    sell_signals = s[s["kalshi_yes_bid"] >= sell_theo]
                    for _, row_data in sell_signals.iterrows():
                        theo_fills.append(("sell", row_data["ts"], row_data["kalshi_yes_bid"]))

            # Plot fill markers on row 1
            if theo_fills:
                buy_fills = [(ts, px) for a, ts, px in theo_fills if a == "buy"]
                sell_fills = [(ts, px) for a, ts, px in theo_fills if a == "sell"]
                label_prefix = "Trade" if use_real_trades else "Signal"
                if buy_fills:
                    fig.add_trace(go.Scatter(
                        x=[f[0] for f in buy_fills], y=[f[1] for f in buy_fills],
                        mode="markers", name=f"Buy {label_prefix}",
                        marker=dict(color="green", size=8, symbol="triangle-up"),
                    ), row=1, col=1)
                if sell_fills:
                    fig.add_trace(go.Scatter(
                        x=[f[0] for f in sell_fills], y=[f[1] for f in sell_fills],
                        mode="markers", name=f"Sell {label_prefix}",
                        marker=dict(color="red", size=8, symbol="triangle-down"),
                    ), row=1, col=1)

            # Compute markouts
            from bisect import bisect_left as _bisect
            snap_times = pd.to_datetime(s["ts"]).values
            snap_bids = s["kalshi_yes_bid"].values
            snap_asks = s["kalshi_yes_ask"].values

            all_markout_vals = []
            markout_data = {sec: [] for sec, _ in markout_intervals}

            for action, fill_ts, fill_price in theo_fills:
                fill_ts_np = pd.Timestamp(fill_ts).to_datetime64()
                for interval_sec, label in markout_intervals:
                    target = fill_ts_np + np.timedelta64(interval_sec, 's')
                    idx = _bisect(snap_times, target)
                    if idx >= len(snap_times):
                        continue
                    if action == "buy":
                        exit_price = snap_bids[idx]
                        markout = (exit_price - fill_price) * 100
                    else:
                        exit_price = snap_asks[idx]
                        markout = (fill_price - exit_price) * 100
                    if exit_price > 0:
                        hover = f"{action} @${fill_price:.2f} → ${exit_price:.2f} ({label})"
                        markout_data[interval_sec].append((fill_ts, markout, hover))
                        all_markout_vals.append(markout)

            # Shared y range
            if all_markout_vals:
                y_max = max(abs(v) for v in all_markout_vals) * 1.15
                markout_yrange = [-y_max, y_max]
            else:
                markout_yrange = None

            for row_idx, (interval_sec, label) in enumerate(markout_intervals, 3):
                data = markout_data[interval_sec]
                if not data:
                    continue
                times = [d[0] for d in data]
                vals = [d[1] for d in data]
                hovers = [d[2] for d in data]
                colors = ["green" if v >= 0 else "red" for v in vals]

                for t, v, c in zip(times, vals, colors):
                    fig.add_trace(go.Scatter(
                        x=[t, t], y=[0, v], mode="lines",
                        line=dict(color=c, width=2),
                        hoverinfo="skip", showlegend=False,
                    ), row=row_idx, col=1)
                fig.add_trace(go.Scatter(
                    x=times, y=vals, mode="markers",
                    marker=dict(size=5, color=colors),
                    hovertext=hovers, hoverinfo="text",
                    showlegend=False,
                ), row=row_idx, col=1)

            # Apply shared y range to markout rows
            if markout_yrange:
                for i in range(3, 7):
                    fig.update_yaxes(range=markout_yrange, row=i, col=1)

            # Draw average lines using add_shape per subplot
            for row_idx, (interval_sec, label) in enumerate(markout_intervals, 3):
                data = markout_data[interval_sec]
                if data:
                    vals = [d[1] for d in data]
                    mean_val = sum(vals) / len(vals)
                    n = len(vals)
                    fig.add_hline(
                        y=mean_val, row=row_idx, col=1,
                        line=dict(color="orange", width=1, dash="dash"),
                        annotation_text=f"avg: {mean_val:+.1f}c (n={n})",
                        annotation_font_color="orange",
                        annotation_font_size=10,
                    )

            # --- Row 7: Cumulative PnL from 5m markouts ---
            # Use 5m markout as the "realized" PnL per trade
            pnl_interval = 300  # 5m
            pnl_data = markout_data.get(pnl_interval, [])
            if pnl_data:
                pnl_sorted = sorted(pnl_data, key=lambda d: str(d[0]))
                pnl_times = [d[0] for d in pnl_sorted]
                pnl_vals = [d[1] for d in pnl_sorted]
                cum_pnl = []
                running = 0
                for v in pnl_vals:
                    running += v
                    cum_pnl.append(running)
                fig.add_trace(go.Scatter(
                    x=pnl_times, y=cum_pnl,
                    mode="lines", name="Cumulative PnL (5m)",
                    line=dict(color="#8b5cf6", width=2),
                    showlegend=False,
                ), row=7, col=1)
                fig.add_hline(y=0, row=7, col=1,
                              line=dict(color="gray", width=1, dash="dot"))

            # Trade count in title
            n_trades = len(theo_fills)
            n_buys = sum(1 for a, _, _ in theo_fills if a == "buy")
            n_sells = n_trades - n_buys
            fig.update_layout(
                title_text=f"Strike {selected_strike:.0f} — {n_trades} trades ({n_buys}B / {n_sells}S)",
                title_font_size=14,
            )

            fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="BTC Price", row=1, col=1, secondary_y=True)
            iv_lo = iv_min if iv_min is not None else 0
            iv_hi = iv_max if iv_max is not None else 100
            fig.update_yaxes(title_text="Implied Vol (%)", range=[iv_lo, iv_hi], row=2, col=1, secondary_y=False)
            fig.update_yaxes(title_text="cents", row=3, col=1)
            fig.update_yaxes(title_text="cents", row=7, col=1)
            fig.update_xaxes(title_text="Time", row=7, col=1)

            fig.update_layout(height=1400, template="plotly_white", hovermode="x unified",
                              uirevision="constant")

            # Store markout data for relayout callback
            store = {}
            for interval_sec, label in markout_intervals:
                store[str(interval_sec)] = [
                    {"ts": str(d[0]), "val": d[1]} for d in markout_data[interval_sec]
                ]

            return fig, store

        @app.callback(
            Output("markout-stats", "children"),
            Input("strike-graph", "relayoutData"),
            Input("markout-store", "data"),
        )
        def update_markout_stats(relayout_data, store):
            if not store:
                return ""

            markout_intervals_local = [(60, "1m"), (300, "5m"), (600, "10m"), (1800, "30m")]

            # Parse visible x range
            x_min = None
            x_max = None
            if relayout_data:
                for key, val in relayout_data.items():
                    if "range[0]" in key and "xaxis" in key:
                        try:
                            x_min = pd.Timestamp(val)
                        except Exception:
                            pass
                    if "range[1]" in key and "xaxis" in key:
                        try:
                            x_max = pd.Timestamp(val)
                        except Exception:
                            pass
                if any("autorange" in k for k in relayout_data):
                    x_min = None
                    x_max = None

            parts = []
            for interval_sec, label in markout_intervals_local:
                data = store.get(str(interval_sec), [])
                if not data:
                    parts.append(f"{label}: no data")
                    continue

                # Filter to visible window
                filtered = data
                if x_min or x_max:
                    filtered = []
                    for d in data:
                        ts = pd.Timestamp(d["ts"])
                        if x_min and ts < x_min:
                            continue
                        if x_max and ts > x_max:
                            continue
                        filtered.append(d)

                if not filtered:
                    parts.append(f"{label}: no data in view")
                    continue

                vals = [d["val"] for d in filtered]
                avg = sum(vals) / len(vals)
                total = sum(vals)
                pos = sum(1 for v in vals if v >= 0)
                neg = len(vals) - pos
                parts.append(f"{label}: avg={avg:+.1f}c  pnl={total:+.0f}c  n={len(vals)}  (+{pos}/-{neg})")

            return "  |  ".join(parts)

        app.run(debug=False)


    @staticmethod
    def implied_sigma(market_price: float, spot: float, strike: float,
                    seconds_to_expiry: float) -> float | None:
        """Back out annualized σ from a binary option's market price.

        Solves N(d2) = market_price for σ.  Assumes r = 0 (fine for
        15-min crypto binaries).  Returns None if the inputs are
        degenerate or no σ in [1e-4, 5.0] matches.
        """
        if not (0 < market_price < 1):
            return None
        if spot <= 0 or strike <= 0 or seconds_to_expiry <= 0:
            return None

        T = seconds_to_expiry / SECONDS_PER_YEAR
        log_sk = math.log(spot / strike)
        SQRT2 = math.sqrt(2.0)

        def price_minus_target(sigma):
            sqrt_T = math.sqrt(T)
            d2 = (log_sk - 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
            n = 0.5 * (1.0 + math.erf(d2 / SQRT2))
            return n - market_price

        try:
            return brentq(price_minus_target, 1e-4, 5.0, xtol=1e-6)
        except ValueError:
            return None  # market_price outside the [σ=1e-4, σ=5] reachable range