"""
Live analysis dashboard for weekly BTC above/below markets.

Graph 1: Market bid/ask vs smoothed-IV theo (selectable span)
Graph 2: Markout bars for init trades (30s, 5m, 15m, 30m)

Reads from analysis/backtesting/data/recorder.db (snapshots + fills).
Auto-refreshes every 30s.

Usage:
    python live_dashboard.py
    # Opens http://127.0.0.1:8050
"""

import math
import sqlite3
import sys
from bisect import bisect_left
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

# Add 4RunnerApp2.0 to path so we can use its KalshiAPI client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "4RunnerApp2.0"))
from kalshi_api import KalshiAPI  # noqa: E402

_api = KalshiAPI()

_norm = NormalDist()

DB_PATH = Path(__file__).parent / "backtesting" / "data" / "recorder.db"  # legacy
DATA_DIR = Path(__file__).parent / "backtesting" / "data"
TZ_LOCAL = "America/Chicago"


def to_ct(series):
    """Convert a UTC timestamp series to Central Time (naive, for plotting)."""
    ts = pd.to_datetime(series, utc=True)
    if hasattr(ts, "dt"):
        return ts.dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)
    return ts.tz_convert(TZ_LOCAL).tz_localize(None)


def event_db_path(event_ticker: str) -> Path:
    return DATA_DIR / f"{event_ticker}.db"
RISK_FREE_RATE = 0.043
SMILE_OTM_PCT = 0.04
MARKOUT_INTERVALS = [
    (10, "10s"),
    (30, "30s"),
    (60, "1m"),
    (300, "5m"),
    (900, "15m"),
    (1800, "30m"),
]

# =============================================================================
# Computation helpers
# =============================================================================

def implied_vol(price: float, spot: float, strike: float,
                T: float, r: float = RISK_FREE_RATE) -> float:
    if price <= 0.01 or price >= 0.99 or spot <= 0 or strike <= 0 or T <= 0:
        return 0.0
    try:
        x = _norm.inv_cdf(price)
        m = math.log(spot / strike) + r * T
        disc = x * x + 2 * m
        if disc < 0:
            return 0.0
        sqrt_disc = math.sqrt(disc)
        u1 = -x + sqrt_disc
        u2 = -x - sqrt_disc
        candidates = [u for u in (u1, u2) if u > 0]
        if not candidates:
            return 0.0
        return min(candidates) / math.sqrt(T)
    except Exception:
        return 0.0


def bs_prob_above(S: float, K: float, sigma: float, T: float,
                  r: float = RISK_FREE_RATE) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return max(min(0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))), 1.0), 0.0)


def compute_smoothed_theo(df: pd.DataFrame, strike: float, span: int) -> pd.DataFrame:
    """Compute smoothed IV theo for a single strike from snapshot data.

    For each timestamp:
      1. Fit quadratic smile on mid_iv of nearby strikes (within 4% of spot)
      2. Evaluate fitted IV at target strike
      3. EWM smooth with given span
      4. Compute theo = N(d2)

    Returns DataFrame with columns: ts, theo, smoothed_iv (indexed by ts).
    """
    if df.empty:
        return pd.DataFrame(columns=["ts", "theo", "smoothed_iv"])

    # Get unique timestamps (sorted, since df is already sorted)
    timestamps = df["ts"].drop_duplicates().values

    # Subsample if too many timestamps to keep fit responsive
    MAX_FITS = 500
    if len(timestamps) > MAX_FITS:
        step = len(timestamps) // MAX_FITS
        timestamps = timestamps[::step]

    fitted_ivs = []
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    # Pre-group once by ts (much faster than repeated df[df["ts"] == ts])
    grouped = df[df["ts"].isin(set(timestamps))].groupby("ts")

    for ts, snap in grouped:
        spot_mid = snap["spot_mid"].iloc[0]
        if spot_mid <= 0:
            fitted_ivs.append((ts, 0.0))
            continue

        otm = (snap["strike"] / spot_mid - 1).abs()
        valid = snap[(otm < SMILE_OTM_PCT) & (snap["mid_iv"] > 0)]

        if len(valid) < 3:
            fitted_ivs.append((ts, 0.0))
            continue

        strikes_np = valid["strike"].values
        ivs_np = valid["mid_iv"].values

        q1, q3 = np.percentile(ivs_np, [25, 75])
        iqr = q3 - q1
        mask = (ivs_np >= q1 - 1.5 * iqr) & (ivs_np <= q3 + 1.5 * iqr)
        if mask.sum() < 3:
            fitted_ivs.append((ts, 0.0))
            continue

        try:
            a, b, c = np.polyfit(strikes_np[mask], ivs_np[mask], 2)
            fitted = a * strike**2 + b * strike + c
            fitted_ivs.append((ts, max(fitted, 0.0)))
        except Exception:
            fitted_ivs.append((ts, 0.0))

    # Sort by ts (groupby doesn't guarantee order)
    fitted_ivs.sort(key=lambda x: x[0])
    ts_arr = [x[0] for x in fitted_ivs]
    iv_arr = [x[1] for x in fitted_ivs]

    ts_series = pd.Series(iv_arr, index=ts_arr)
    ts_series = ts_series.replace(0.0, np.nan)
    smoothed = ts_series.ewm(span=span, min_periods=1).mean()

    timestamps = ts_arr
    ts_data = df.drop_duplicates("ts").set_index("ts")[["T", "spot_mid"]].reindex(timestamps)

    result = pd.DataFrame({
        "ts": timestamps,
        "smoothed_iv": smoothed.values,
        "T": ts_data["T"].values,
        "spot_mid": ts_data["spot_mid"].values,
    })
    result["theo"] = result.apply(
        lambda r: bs_prob_above(r["spot_mid"], strike, r["smoothed_iv"], r["T"])
        if r["smoothed_iv"] > 0 and r["T"] > 0 else np.nan, axis=1
    )
    return result


def compute_trade_markouts(fills: pd.DataFrame, trades_list: list) -> dict:
    """Compute "best-exit" markouts using public trade prices.

    For a buy fill, exit = highest yes-taker trade price in window
                         (someone willing to buy yes at that price = we'd sell to them).
    For a sell fill, exit = lowest no-taker trade price in window
                          (someone willing to sell yes at that price = we'd buy from them).
    """
    results = {sec: [] for sec, _ in MARKOUT_INTERVALS}
    if fills.empty or not trades_list:
        return results

    # Pre-parse trade timestamps once
    parsed_trades = []
    for t in trades_list:
        ts = pd.Timestamp(t["ts"])
        if ts.tzinfo:
            ts = ts.tz_convert(TZ_LOCAL).tz_localize(None)
        parsed_trades.append((ts, t["px"], t.get("side", "")))
    parsed_trades.sort(key=lambda x: x[0])
    trade_times = [t[0] for t in parsed_trades]

    for _, fill in fills.iterrows():
        fill_ts = to_ct(pd.Series([fill["ts"]])).iloc[0]
        fill_price = fill["price"]
        action = fill["action"]

        for interval_sec, label in MARKOUT_INTERVALS:
            target = fill_ts + pd.Timedelta(seconds=interval_sec)
            # Find trades in (fill_ts, target]
            start_idx = bisect_left(trade_times, fill_ts)
            end_idx = bisect_left(trade_times, target)
            window = parsed_trades[start_idx:end_idx]
            if not window:
                continue

            if action == "buy":
                # Best exit = highest yes-taker price (someone bought yes from us)
                yes_prices = [px for _, px, side in window if side == "yes"]
                if not yes_prices:
                    continue
                exit_price = max(yes_prices)
                markout = (exit_price - fill_price) * 100
            else:
                # Best exit = lowest no-taker price (someone sold yes to us)
                no_prices = [px for _, px, side in window if side == "no"]
                if not no_prices:
                    continue
                exit_price = min(no_prices)
                markout = (fill_price - exit_price) * 100

            hover = (f"{fill['strike']:.0f} {action} @${fill_price:.2f} "
                     f"→ best trade ${exit_price:.2f} ({label})")
            results[interval_sec].append((fill_ts, markout, hover))

    return results


def compute_markouts(fills: pd.DataFrame, snapshots: pd.DataFrame) -> dict:
    """Compute markouts for each fill at each interval.

    Returns {interval_sec: [(ts, markout_cents, hover), ...]}.
    """
    results = {sec: [] for sec, _ in MARKOUT_INTERVALS}
    if fills.empty or snapshots.empty:
        return results

    for _, fill in fills.iterrows():
        ticker = fill["ticker"]
        fill_ts = to_ct(pd.Series([fill["ts"]])).iloc[0]
        fill_price = fill["price"]
        action = fill["action"]

        # Get snapshots for this ticker, sorted by time
        tk_snaps = snapshots[snapshots["ticker"] == ticker].sort_values("ts")
        if tk_snaps.empty:
            continue
        snap_times = to_ct(tk_snaps["ts"])
        snap_bids = tk_snaps["kalshi_yes_bid"].values
        snap_asks = tk_snaps["kalshi_yes_ask"].values

        for interval_sec, label in MARKOUT_INTERVALS:
            target = fill_ts + pd.Timedelta(seconds=interval_sec)
            idx = bisect_left(snap_times.values, target.to_datetime64())
            if idx >= len(snap_times):
                continue

            if action == "buy":
                exit_price = snap_bids[idx]  # what you'd sell at
                markout = (exit_price - fill_price) * 100  # cents
            else:
                exit_price = snap_asks[idx]  # what you'd buy at
                markout = (fill_price - exit_price) * 100  # cents

            if exit_price > 0:
                hover = (f"{fill['strike']:.0f} {action} @${fill_price:.2f} "
                         f"→ ${exit_price:.2f} ({label})")
                results[interval_sec].append((fill_ts, markout, hover))

    return results


# =============================================================================
# Dash App
# =============================================================================

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#0b0f19", "color": "#c8cdd5",
           "fontFamily": "monospace", "padding": "15px",
           "minHeight": "100vh"},
    children=[
        html.H2("Live Analysis — Weekly BTC",
                style={"color": "#facc15", "marginBottom": "10px"}),

        html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "15px",
                        "alignItems": "center"}, children=[
            html.Label("Event:"),
            dcc.Dropdown(id="event-dropdown", style={"width": "220px",
                         "backgroundColor": "#141923", "color": "#000"}),
            html.Label("Strike:"),
            dcc.Dropdown(id="strike-dropdown", style={"width": "150px",
                         "backgroundColor": "#141923", "color": "#000"}),
            html.Label("Span:"),
            dcc.Dropdown(id="span-dropdown",
                         options=[{"label": str(s), "value": s}
                                  for s in [10, 20, 30, 60, 100]],
                         value=60,
                         style={"width": "100px",
                                "backgroundColor": "#141923", "color": "#000"}),
            dcc.Checklist(
                id="show-trades",
                options=[{"label": " Show Trades", "value": "show"}],
                value=["show"],
                style={"color": "#ffffff", "marginLeft": "20px",
                       "fontSize": "14px"},
                labelStyle={"color": "#ffffff"},
            ),
            dcc.Checklist(
                id="show-orders",
                options=[{"label": " Show Orders", "value": "show"}],
                value=["show"],
                style={"color": "#ffffff", "marginLeft": "10px",
                       "fontSize": "14px"},
                labelStyle={"color": "#ffffff"},
            ),
            dcc.Checklist(
                id="show-legend",
                options=[{"label": " Show Legend", "value": "show"}],
                value=["show"],
                style={"color": "#ffffff", "marginLeft": "10px",
                       "fontSize": "14px"},
                labelStyle={"color": "#ffffff"},
            ),
            html.Label("Markouts:",
                       style={"marginLeft": "20px", "color": "#ffffff"}),
            dcc.RadioItems(
                id="markout-mode",
                options=[
                    {"label": " BBO", "value": "bbo"},
                    {"label": " Trade", "value": "trade"},
                    {"label": " Both", "value": "both"},
                ],
                value="both",
                inline=True,
                style={"color": "#ffffff", "marginLeft": "8px"},
                labelStyle={"color": "#ffffff", "marginRight": "10px"},
            ),
        ]),

        dcc.Graph(id="combined-graph",
                  style={"height": "900px"},
                  config={"displayModeBar": True}),

        html.Div(id="markout-stats",
                 style={"padding": "8px 12px", "fontFamily": "monospace",
                        "fontSize": "13px", "color": "#c8cdd5",
                        "backgroundColor": "#0b0f19",
                        "borderTop": "1px solid #1e2736"}),

        dcc.Interval(id="refresh", interval=30_000, n_intervals=0),

        # Cache: {ticker: [trades]}
        dcc.Store(id="trades-store"),
    ]
)


def load_data():
    """Load snapshots and fills from the recorder DB."""
    if not DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(str(DB_PATH))
    try:
        snaps = pd.read_sql("SELECT * FROM snapshots ORDER BY ts", conn)
        fills = pd.read_sql("SELECT * FROM fills ORDER BY ts", conn)
    except Exception:
        snaps = pd.DataFrame()
        fills = pd.DataFrame()
    conn.close()
    return snaps, fills


def load_events():
    """Lightweight: list event files in data/ and read each event's metadata."""
    rows = []
    for db_file in DATA_DIR.glob("*.db"):
        if db_file.name == "recorder.db":
            continue
        event_ticker = db_file.stem
        try:
            conn = sqlite3.connect(str(db_file))
            df = pd.read_sql(
                "SELECT MAX(close_time) AS close_time, MAX(spot_mid) AS spot_mid "
                "FROM snapshots", conn
            )
            conn.close()
            if not df.empty:
                rows.append({
                    "event_ticker": event_ticker,
                    "close_time": df["close_time"].iloc[0],
                    "spot_mid": df["spot_mid"].iloc[0],
                })
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_strikes_for_event(event_ticker):
    """Get distinct strikes + most recent spot for an event."""
    if not event_ticker:
        return pd.DataFrame()
    db_file = event_db_path(event_ticker)
    if not db_file.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_file))
    try:
        df = pd.read_sql("""
            SELECT DISTINCT strike,
                   (SELECT spot_mid FROM snapshots ORDER BY ts DESC LIMIT 1) AS spot_mid
            FROM snapshots
        """, conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def load_event_data(event_ticker, strike=None, hours: float = 24.0):
    """Load snapshots (last `hours`) + all fills for a specific event.

    Snapshots can be 800k+ rows over a full day at 5s cadence × N strikes;
    truncate to a recent window for the dashboard's chart needs.  Fills
    are sparse so we keep all of them.
    """
    if not event_ticker:
        return pd.DataFrame(), pd.DataFrame()
    db_file = event_db_path(event_ticker)
    if not db_file.exists():
        return pd.DataFrame(), pd.DataFrame()
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    cutoff_iso = (_dt.now(tz=_tz.utc) - _td(hours=hours)).isoformat()
    conn = sqlite3.connect(str(db_file))
    try:
        snaps = pd.read_sql(
            "SELECT * FROM snapshots WHERE ts >= ? ORDER BY ts",
            conn, params=[cutoff_iso],
        )
        fills = pd.read_sql("SELECT * FROM fills ORDER BY ts", conn)
    except Exception:
        snaps = pd.DataFrame()
        fills = pd.DataFrame()
    conn.close()
    return snaps, fills


def _events_table_exists(conn) -> bool:
    """True if the recorder DB has the events firehose table."""
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='events'"
        ).fetchone()
        return row is not None
    except Exception:
        return False


_INDEX_ENSURED: set = set()  # remember which DBs we've already indexed


def _ensure_event_index(conn, db_path: str):
    """Create the optimal composite index on first use of a DB.

    `WHERE event_type=? AND ticker=? AND ts_us>=?` is hit by every chart
    loader; (event_type, ticker, ts_us) is the matching composite index.
    SQLite picks it automatically once it exists.  Idempotent.
    """
    if db_path in _INDEX_ENSURED:
        return
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_ticker_ts "
            "ON events (event_type, ticker, ts_us)"
        )
        conn.commit()
        _INDEX_ENSURED.add(db_path)
    except Exception:
        pass


DEFAULT_LOOKBACK_HOURS = 24  # cover the full event session
# Theo events publish on every value change (often 5–50/sec), which can
# exceed 200k rows in a busy session.  Pulling all of them via SQL
# json_extract takes 10s+ so we subsample at the SQL layer:
#   - SUBSAMPLE_TARGET rows spread evenly across the full lookback window
#   - one row per (window_seconds / SUBSAMPLE_TARGET)-second bucket
# That gives full coverage with a bounded cost regardless of update rate.
SUBSAMPLE_TARGET = 8000      # rows per loader — same target whether the
                              # window is full-event (fast subsample) or a
                              # tight zoom (denser, near-tick-level detail).
# Orders are sparse so they don't need subsampling — keep all of them.
MAX_ORDER_ROWS = 30000


def _since_us(hours: float) -> int:
    """Microseconds-since-epoch threshold for `now - hours`."""
    import time as _t
    return int((_t.time() - hours * 3600) * 1_000_000)


def _zoom_window_us(relayout_data) -> tuple | None:
    """Parse plotly relayoutData and return (since_us, until_us) of the
    visible x-range, or None if there's no zoom (autorange or no data).
    Plotly emits xaxis range timestamps as the displayed (CT-naive)
    strings — we convert via the local tz back to UTC microseconds."""
    if not relayout_data:
        return None
    if any("autorange" in k for k in relayout_data):
        return None
    x0_raw = None
    x1_raw = None
    for key, val in relayout_data.items():
        if "range[0]" in key and "xaxis" in key:
            x0_raw = val
        elif "range[1]" in key and "xaxis" in key:
            x1_raw = val
    if not (x0_raw and x1_raw):
        return None
    try:
        # The chart x-axis is in TZ_LOCAL-naive timestamps.  Localize then
        # convert to UTC, then to microseconds.
        x0 = pd.Timestamp(x0_raw).tz_localize(TZ_LOCAL).tz_convert("UTC")
        x1 = pd.Timestamp(x1_raw).tz_localize(TZ_LOCAL).tz_convert("UTC")
        return int(x0.value / 1000), int(x1.value / 1000)
    except Exception:
        return None


def load_book_events(event_ticker: str, ticker: str,
                     hours: float = DEFAULT_LOOKBACK_HOURS,
                     window_us: tuple | None = None) -> pd.DataFrame:
    """Load TOB changes for a market.  Window is either
    `(since_us, until_us)` (used for zoom-aware reloads) or `last hours`.
    Subsamples to SUBSAMPLE_TARGET rows via per-bucket MAX(id).
    """
    if not event_ticker or not ticker:
        return pd.DataFrame()
    db_file = event_db_path(event_ticker)
    if not db_file.exists():
        return pd.DataFrame()
    if window_us is not None:
        since, until = window_us
        span_us = max(until - since, 1)
    else:
        since = _since_us(hours)
        until = None
        span_us = int(hours * 3600 * 1_000_000)
    bucket_us = max(span_us // SUBSAMPLE_TARGET, 1)
    conn = sqlite3.connect(str(db_file))
    try:
        if not _events_table_exists(conn):
            return pd.DataFrame()
        _ensure_event_index(conn, str(db_file))
        if until is None:
            inner = ("SELECT MAX(id) FROM events "
                     "WHERE event_type='book_tob' AND ticker=? AND ts_us >= ? "
                     "GROUP BY ts_us / ?")
            params = [ticker, since, bucket_us]
        else:
            inner = ("SELECT MAX(id) FROM events "
                     "WHERE event_type='book_tob' AND ticker=? "
                     "AND ts_us >= ? AND ts_us <= ? "
                     "GROUP BY ts_us / ?")
            params = [ticker, since, until, bucket_us]
        df = pd.read_sql(
            "SELECT ts_us, "
            "       json_extract(payload,'$.yes_bid')  AS yes_bid, "
            "       json_extract(payload,'$.yes_ask')  AS yes_ask, "
            "       json_extract(payload,'$.bid_size') AS bid_size, "
            "       json_extract(payload,'$.ask_size') AS ask_size "
            f"FROM events WHERE id IN ({inner}) ORDER BY ts_us",
            conn, params=params,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_us"], unit="us", utc=True)
    return df[["ts", "yes_bid", "yes_ask", "bid_size", "ask_size"]]


def load_order_events(event_ticker: str, ticker: str,
                      hours: float = DEFAULT_LOOKBACK_HOURS) -> pd.DataFrame:
    """Load order state changes for a market — last `hours` only by default.

    Uses SQL-side json_extract to avoid Python-loop JSON decoding.
    """
    if not event_ticker or not ticker:
        return pd.DataFrame()
    db_file = event_db_path(event_ticker)
    if not db_file.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_file))
    try:
        if not _events_table_exists(conn):
            return pd.DataFrame()
        _ensure_event_index(conn, str(db_file))
        df = pd.read_sql(
            "SELECT ts_us, "
            "       json_extract(payload,'$.order_id')           AS order_id, "
            "       json_extract(payload,'$.status')             AS status, "
            "       json_extract(payload,'$.action')             AS action, "
            "       json_extract(payload,'$.side')               AS side, "
            "       json_extract(payload,'$.is_yes')             AS is_yes, "
            "       json_extract(payload,'$.yes_price_dollars')  AS price, "
            "       json_extract(payload,'$.initial_count_fp')   AS init_cnt, "
            "       json_extract(payload,'$.remaining_count_fp') AS rem_cnt, "
            "       json_extract(payload,'$.fill_count_fp')      AS fill_count, "
            "       json_extract(payload,'$.client_order_id')    AS client_order_id, "
            "       json_extract(payload,'$.is_taker')           AS is_taker "
            "FROM events "
            "WHERE event_type='order' AND ticker=? AND ts_us >= ? "
            "ORDER BY ts_us DESC LIMIT ?",
            conn, params=[ticker, _since_us(hours), MAX_ORDER_ROWS],
        )
        df = df.iloc[::-1].reset_index(drop=True)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts_us"], unit="us", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["count"] = pd.to_numeric(
        df["init_cnt"].fillna(df["rem_cnt"]), errors="coerce").fillna(0.0)
    df["fill_count"] = pd.to_numeric(df["fill_count"], errors="coerce").fillna(0.0)
    df["is_taker"] = df["is_taker"].fillna(0).astype(bool)
    df["client_order_id"] = df["client_order_id"].fillna("")
    df["status"] = df["status"].fillna("")
    df["action"] = df["action"].fillna("")
    df["side"] = df["side"].fillna("")
    df["order_id"] = df["order_id"].fillna("")

    def _tag(coid: str) -> str:
        if coid.startswith("phase3t_"): return "phase3t"
        if coid.startswith("phase3d_"): return "phase3d"
        if coid.startswith("phase3_"):  return "phase3"
        if coid.startswith("init_"):    return "init"
        if coid.startswith("flat_"):    return "flat"
        return "?"
    df["tag"] = df["client_order_id"].map(_tag)
    # `is_yes` from json_extract returns 0/1 (or None) — keep as-is so
    # downstream renderers can `.fillna(0).astype(bool)`.
    df["is_yes"] = df["is_yes"].fillna(0).astype(int)
    return df[["ts", "order_id", "status", "action", "side", "is_yes",
               "price", "count", "fill_count", "client_order_id",
               "tag", "is_taker"]]


def first_seen_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce an order events dataframe to one row per order_id, keeping
    the earliest event (the placement/acceptance).  Useful for showing
    'where orders were placed' as opposed to the full lifecycle."""
    if orders_df.empty or "order_id" not in orders_df.columns:
        return orders_df
    return (orders_df.sort_values("ts")
                     .drop_duplicates(subset=["order_id"], keep="first"))


def load_theo_events(event_ticker: str, ticker: str,
                     hours: float = DEFAULT_LOOKBACK_HOURS,
                     window_us: tuple | None = None) -> pd.DataFrame:
    """Load theo events published by the app, subsampled to SUBSAMPLE_TARGET
    rows.  Window is `(since_us, until_us)` for zoom reloads, otherwise
    last `hours`."""
    if not event_ticker or not ticker:
        return pd.DataFrame()
    db_file = event_db_path(event_ticker)
    if not db_file.exists():
        return pd.DataFrame()
    if window_us is not None:
        since, until = window_us
        span_us = max(until - since, 1)
    else:
        since = _since_us(hours)
        until = None
        span_us = int(hours * 3600 * 1_000_000)
    bucket_us = max(span_us // SUBSAMPLE_TARGET, 1)
    conn = sqlite3.connect(str(db_file))
    try:
        if not _events_table_exists(conn):
            return pd.DataFrame()
        _ensure_event_index(conn, str(db_file))
        if until is None:
            inner = ("SELECT MAX(id) FROM events "
                     "WHERE event_type='theo' AND ticker=? AND ts_us >= ? "
                     "GROUP BY ts_us / ?")
            params = [ticker, since, bucket_us]
        else:
            inner = ("SELECT MAX(id) FROM events "
                     "WHERE event_type='theo' AND ticker=? "
                     "AND ts_us >= ? AND ts_us <= ? "
                     "GROUP BY ts_us / ?")
            params = [ticker, since, until, bucket_us]
        df = pd.read_sql(
            "SELECT ts_us, "
            "       json_extract(payload,'$.theo')         AS theo, "
            "       json_extract(payload,'$.smoothed_iv')  AS smoothed_iv, "
            "       json_extract(payload,'$.kalshi_bid')   AS kalshi_bid, "
            "       json_extract(payload,'$.kalshi_ask')   AS kalshi_ask, "
            "       json_extract(payload,'$.spot')         AS spot "
            f"FROM events WHERE id IN ({inner}) ORDER BY ts_us",
            conn, params=params,
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_us"], unit="us", utc=True)
    return df[["ts", "theo", "smoothed_iv", "kalshi_bid", "kalshi_ask", "spot"]]


@app.callback(
    Output("event-dropdown", "options"),
    Output("event-dropdown", "value"),
    Input("refresh", "n_intervals"),
    State("event-dropdown", "value"),
)
def refresh_events(n, current_event):
    ev_df = load_events()
    if ev_df.empty:
        return [], None
    events = sorted(ev_df["event_ticker"].unique())
    options = [{"label": e, "value": e} for e in events]

    # Keep current selection if still valid
    if current_event and current_event in events:
        return options, current_event

    # Default to Friday expiry
    best = events[-1]
    ev_lookup = {row["event_ticker"]: row["close_time"] for _, row in ev_df.iterrows()}
    for e in events:
        ct = ev_lookup.get(e, "") or ""
        if ct:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(ct.replace("Z", "+00:00"))
                if dt.weekday() == 4:
                    best = e
            except Exception:
                pass
    return options, best


@app.callback(
    Output("strike-dropdown", "options"),
    Output("strike-dropdown", "value"),
    Input("event-dropdown", "value"),
    Input("refresh", "n_intervals"),
    State("strike-dropdown", "value"),
)
def refresh_strikes(event, n, current_strike):
    if not event:
        return [], None
    strikes_df = load_strikes_for_event(event)
    if strikes_df.empty:
        return [], None
    strikes = sorted(strikes_df["strike"].unique())

    # Count init fills per strike (light query, per-event DB)
    init_counts = {}
    db_file = event_db_path(event)
    if db_file.exists():
        try:
            conn = sqlite3.connect(str(db_file))
            cnt_df = pd.read_sql("""
                SELECT strike, COUNT(*) AS n FROM fills
                WHERE client_order_id LIKE 'init_%'
                GROUP BY strike
            """, conn)
            conn.close()
            init_counts = dict(zip(cnt_df["strike"], cnt_df["n"]))
        except Exception:
            pass

    options = []
    for s in strikes:
        count = init_counts.get(s, 0)
        label = f"${s:,.0f} ({count})" if count > 0 else f"${s:,.0f}"
        options.append({"label": label, "value": s})

    # Keep current selection if still valid
    if current_strike and current_strike in strikes:
        return options, current_strike

    spot = strikes_df["spot_mid"].iloc[0] if not strikes_df.empty else 0
    if spot > 0:
        default = min(strikes, key=lambda s: abs(s - spot))
    else:
        default = strikes[len(strikes) // 2]
    return options, default


@app.callback(
    Output("trades-store", "data"),
    Input("event-dropdown", "value"),
    Input("strike-dropdown", "value"),
)
def fetch_trades(event, strike):
    """Fetch all public trades for the selected event+strike's ticker."""
    if not event or strike is None:
        raise PreventUpdate

    # Look up the ticker from the per-event DB (one ticker per strike)
    db_file = event_db_path(event)
    if not db_file.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_file))
        cur = conn.execute(
            "SELECT ticker FROM snapshots WHERE strike = ? LIMIT 1", (strike,)
        )
        row = cur.fetchone()
        conn.close()
    except Exception:
        return {}
    if not row:
        return {}
    ticker = row[0]

    try:
        trades = _api.get_trades(ticker, limit=5000)
    except Exception as e:
        print(f"[Dashboard] Failed to fetch trades for {ticker}: {e}")
        return {}

    # Compact storage — only fields we need
    return {
        "ticker": ticker,
        "trades": [
            {
                "ts": t["created_time"],
                "px": float(t.get("yes_price_dollars", 0)),
                "size": float(t.get("count_fp", t.get("count", 0)) or 0),
                "side": t.get("taker_side", ""),
            }
            for t in trades
            if float(t.get("yes_price_dollars", 0)) > 0
        ],
    }


@app.callback(
    Output("combined-graph", "figure"),
    Input("event-dropdown", "value"),
    Input("strike-dropdown", "value"),
    Input("span-dropdown", "value"),
    Input("refresh", "n_intervals"),
    Input("show-trades", "value"),
    Input("show-orders", "value"),
    Input("show-legend", "value"),
    Input("markout-mode", "value"),
    Input("combined-graph", "relayoutData"),
    State("trades-store", "data"),
)
def update_graphs(event, strike, span, n, show_trades, show_orders,
                  show_legend_val, markout_mode, relayout_data, trades_data):
    # Zoom-aware reload: when the user has zoomed in, re-query the data
    # within the visible x-range at full subsample target (so a 5-min zoom
    # gets ~tick-level detail, not 5 min worth of subsampled-from-24h).
    zoom_window = _zoom_window_us(relayout_data)
    dark_bg = "#0b0f19"
    grid_color = "#1e2736"

    n_rows = 1 + len(MARKOUT_INTERVALS)
    row_titles = [
        f"Market vs Theo — ${strike:,.0f}" if strike else "Market vs Theo",
    ] + [label for _, label in MARKOUT_INTERVALS]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=[3] + [1] * len(MARKOUT_INTERVALS),
        subplot_titles=row_titles,
        vertical_spacing=0.03,
    )
    # Toggle legend.  Include the legend state in `uirevision` so plotly
    # actually re-applies the layout when the toggle flips (uirevision
    # otherwise preserves the user's last-known UI state across updates,
    # which can suppress programmatic showlegend changes).
    show_legend = bool(show_legend_val and "show" in show_legend_val)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=dark_bg, plot_bgcolor=dark_bg,
        margin=dict(l=50, r=120, t=40, b=30),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(20, 25, 35, 0.92)",
            bordercolor="#facc15",
            borderwidth=1,
            font=dict(color="#ffffff", size=12),
            x=1.01, y=1.0,
            xanchor="left", yanchor="top",
            orientation="v",
            itemsizing="constant",
        ),
        uirevision=f"{event}|{strike}|leg{int(show_legend)}",
        hovermode="x unified",
    )

    snaps, fills = load_event_data(event) if event else (pd.DataFrame(), pd.DataFrame())

    if snaps.empty or strike is None or event is None:
        return fig

    # snaps and fills are already filtered to the selected event by load_event_data

    # Filter snapshots for selected strike
    strike_snaps = snaps[snaps["strike"] == strike].copy()
    if strike_snaps.empty:
        return fig

    # Resolve the ticker for this strike — needed for the events table loaders
    ticker = strike_snaps["ticker"].iloc[0] if "ticker" in strike_snaps.columns else None

    # Bid/ask line uses the snapshots table (5s cadence — covers the full
    # event session smoothly).  The events table's tick-level book_tob
    # stream is preserved for forensic queries but isn't needed on the
    # main chart and would either dominate plot density or be window-
    # capped too aggressively.
    ts = to_ct(strike_snaps["ts"])
    bid_series = strike_snaps["kalshi_yes_bid"]
    ask_series = strike_snaps["kalshi_yes_ask"]

    # Row 1: Market bid/ask — WebGL renderer for the high-density lines.
    fig.add_trace(go.Scattergl(
        x=ts, y=bid_series,
        mode="lines", name="Bid", line=dict(color="#22c55e", width=1),
        hovertemplate="Bid: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=ts, y=ask_series,
        mode="lines", name="Ask", line=dict(color="#ef4444", width=1),
        hovertemplate="Ask: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Public trades (toggleable)
    if show_trades and "show" in show_trades and trades_data:
        trades_list = trades_data.get("trades", [])
        if trades_list:
            buys, sells = [], []
            for t in trades_list:
                row = (t["ts"], t["px"], t.get("size", 0))
                if t.get("side") == "yes":
                    buys.append(row)  # taker bought yes
                else:
                    sells.append(row)
            if buys:
                fig.add_trace(go.Scatter(
                    x=to_ct(pd.Series([b[0] for b in buys])),
                    y=[b[1] for b in buys],
                    mode="markers", name="Trade (yes taker)",
                    marker=dict(symbol="triangle-up", color="#06b6d4",
                                size=10, line=dict(color="#0891b2", width=1)),
                    customdata=[b[2] for b in buys],
                    hovertemplate=("Trade (yes taker): $%{y:.2f} "
                                   "x%{customdata:.2f}<extra></extra>"),
                ), row=1, col=1)
            if sells:
                fig.add_trace(go.Scatter(
                    x=to_ct(pd.Series([s[0] for s in sells])),
                    y=[s[1] for s in sells],
                    mode="markers", name="Trade (no taker)",
                    marker=dict(symbol="triangle-down", color="#f97316",
                                size=10, line=dict(color="#c2410c", width=1)),
                    customdata=[s[2] for s in sells],
                    hovertemplate=("Trade (no taker): $%{y:.2f} "
                                   "x%{customdata:.2f}<extra></extra>"),
                ), row=1, col=1)

    # Theo line — prefer the app's published theos (exact match to what
    # the strategy used).  Fall back to client-side smile-fit if the
    # events table has no theo events yet.
    published_theo = (load_theo_events(event, ticker, window_us=zoom_window)
                      if ticker else pd.DataFrame())
    if not published_theo.empty:
        fig.add_trace(go.Scattergl(
            x=to_ct(published_theo["ts"]), y=published_theo["theo"],
            mode="lines", name="Theo (app)",
            line=dict(color="#8b5cf6", width=2),
            hovertemplate="Theo: $%{y:.3f}<extra></extra>",
        ), row=1, col=1)
    else:
        theo_df = compute_smoothed_theo(snaps, strike, span)
        if not theo_df.empty and theo_df["theo"].notna().any():
            theo_ts = to_ct(theo_df["ts"])
            fig.add_trace(go.Scattergl(
                x=theo_ts, y=theo_df["theo"],
                mode="lines", name=f"Theo (span={span})",
                line=dict(color="#8b5cf6", width=2),
                hovertemplate="Theo: $%{y:.3f}<extra></extra>",
            ), row=1, col=1)

    # Fill markers — ALL fills (init / flat / phase3), selected strike.
    # init fills are still used separately for markout computation below.
    all_fills = fills[fills["strike"] == strike] if not fills.empty else pd.DataFrame()
    init_fills = (
        all_fills[all_fills["client_order_id"].str.startswith("init_", na=False)]
        if not all_fills.empty and "client_order_id" in all_fills.columns
        else pd.DataFrame()
    )

    def _fill_tag(c: str) -> str:
        c = c or ""
        if c.startswith("phase3t_"): return "phase3t"
        if c.startswith("phase3d_"): return "phase3d"
        if c.startswith("phase3_"):  return "phase3"
        if c.startswith("init_"):    return "init"
        if c.startswith("flat_"):    return "flat"
        return "?"

    if not all_fills.empty:
        all_fills = all_fills.copy()
        all_fills["tag"] = all_fills["client_order_id"].fillna("").map(_fill_tag)
        # Filled triangles (buy = up green, sell = down red); shape size
        # bigger than placement markers (10 vs 8) and filled vs open so
        # they read clearly even if a placement marker sits at the same spot.
        for action, sym, color, name in (
            ("buy",  "triangle-up",   "#22c55e", "My Buy (filled)"),
            ("sell", "triangle-down", "#ef4444", "My Sell (filled)"),
        ):
            sub = all_fills[all_fills["action"] == action]
            if sub.empty:
                continue
            customdata = list(zip(sub["count"], sub["tag"]))
            fig.add_trace(go.Scatter(
                x=to_ct(sub["ts"]), y=sub["price"],
                mode="markers", name=name,
                marker=dict(symbol=sym, size=10, color=color),
                customdata=customdata,
                hovertemplate=(
                    f"My {action.upper()}: $%{{y:.2f}} "
                    "x%{customdata[0]:.0f} [%{customdata[1]}]<extra></extra>"
                ),
            ), row=1, col=1)

    # Order placement markers (toggleable) — show every order_id we
    # placed at its acceptance price/time, regardless of whether it filled.
    # Useful for spotting cancelled/rejected orders that fills don't capture.
    #
    # Shape encodes side: triangle-up = bid yes (buying), triangle-down =
    # ask yes (selling).  Color encodes intent (init/flat/phase3).
    if (show_orders and "show" in show_orders and ticker
            and not strike_snaps.empty):
        orders_all = load_order_events(event, ticker)  # orders are sparse, no zoom reload
        if not orders_all.empty:
            placements = first_seen_orders(orders_all)
            placements = placements[placements["price"] > 0]
            if not placements.empty:
                # is_yes True = buying yes (bid yes); False = selling yes (ask yes)
                placements["is_bid"] = placements["is_yes"].fillna(0).astype(bool)
                tag_colors = {
                    "init":     "#3b82f6",
                    "flat":     "#f59e0b",
                    "phase3":   "#ef4444",
                    "phase3t":  "#ef4444",
                    "phase3d":  "#ef4444",
                }
                for tag, color in tag_colors.items():
                    for is_bid, dir_label, symbol in (
                        (True,  "bid yes", "triangle-up-open"),
                        (False, "ask yes", "triangle-down-open"),
                    ):
                        sub = placements[(placements["tag"] == tag)
                                         & (placements["is_bid"] == is_bid)]
                        if sub.empty:
                            continue
                        customdata = list(zip(sub["count"], sub["tag"],
                                              sub["status"], sub["order_id"]))
                        fig.add_trace(go.Scatter(
                            x=to_ct(sub["ts"]), y=sub["price"],
                            mode="markers",
                            name=f"{dir_label} ({tag})",
                            marker=dict(symbol=symbol, size=10,
                                        color=color,
                                        line=dict(width=1.5, color=color)),
                            customdata=customdata,
                            hovertemplate=(
                                f"{dir_label} [{tag}]: $%{{y:.2f}} "
                                "x%{customdata[0]:.0f} %{customdata[2]}"
                                "<br>id=%{customdata[3]}<extra></extra>"
                            ),
                        ), row=1, col=1)

    # Rows 2-5: Markouts (selected strike only)
    markouts = compute_markouts(init_fills, snaps)

    # Trade-based markouts (from public trades, if loaded)
    trade_markouts = {}
    if trades_data and "trades" in trades_data:
        trade_markouts = compute_trade_markouts(init_fills, trades_data["trades"])

    # Find global y range across all markout intervals (both BBO and trade-based)
    all_markout_vals = []
    for interval_sec, _ in MARKOUT_INTERVALS:
        all_markout_vals.extend([d[1] for d in markouts[interval_sec]])
        all_markout_vals.extend([d[1] for d in trade_markouts.get(interval_sec, [])])
    if all_markout_vals:
        y_max = max(abs(v) for v in all_markout_vals) * 1.15
        markout_yrange = [-y_max, y_max]
    else:
        markout_yrange = None

    show_bbo = markout_mode in ("bbo", "both")
    show_trade = markout_mode in ("trade", "both")

    for row_idx, (interval_sec, _) in enumerate(MARKOUT_INTERVALS, 2):
        # BBO-based markouts (green/red stems)
        if show_bbo:
            data = markouts[interval_sec]
            if data:
                times = [d[0] for d in data]
                vals = [d[1] for d in data]
                hovers = [d[2] for d in data]
                colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]

                for t, v, h, c in zip(times, vals, hovers, colors):
                    fig.add_trace(go.Scatter(
                        x=[t, t], y=[0, v],
                        mode="lines", line=dict(color=c, width=2),
                        hoverinfo="skip", showlegend=False,
                    ), row=row_idx, col=1)
                fig.add_trace(go.Scatter(
                    x=times, y=vals,
                    mode="markers",
                    marker=dict(size=6, color=colors),
                    hovertext=hovers, hoverinfo="text",
                    name="BBO markout" if row_idx == 2 else None,
                    showlegend=(row_idx == 2),
                ), row=row_idx, col=1)

        # Trade-based markouts (cyan/orange diamond markers)
        if show_trade:
            t_data = trade_markouts.get(interval_sec, [])
            if t_data:
                t_times = [d[0] for d in t_data]
                t_vals = [d[1] for d in t_data]
                t_hovers = [d[2] for d in t_data]
                t_colors = ["#06b6d4" if v >= 0 else "#f97316" for v in t_vals]

                fig.add_trace(go.Scatter(
                    x=t_times, y=t_vals,
                    mode="markers",
                    marker=dict(symbol="diamond", size=8, color=t_colors,
                                line=dict(color="#ffffff", width=0.5)),
                    hovertext=t_hovers, hoverinfo="text",
                    name="Trade-based markout" if row_idx == 2 else None,
                    showlegend=(row_idx == 2),
                ), row=row_idx, col=1)

    # Style all axes
    for i in range(1, n_rows + 1):
        fig.update_xaxes(gridcolor=grid_color, row=i, col=1)
        fig.update_yaxes(gridcolor=grid_color, row=i, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="cents", row=2, col=1)
    # Shared y scale for all markout rows
    if markout_yrange:
        for i in range(2, n_rows + 1):
            fig.update_yaxes(range=markout_yrange, row=i, col=1)

    return fig


@app.callback(
    Output("markout-stats", "children"),
    Input("event-dropdown", "value"),
    Input("strike-dropdown", "value"),
    Input("refresh", "n_intervals"),
    Input("markout-mode", "value"),
    Input("combined-graph", "relayoutData"),
    State("trades-store", "data"),
)
def update_markout_stats(event, strike, n, markout_mode, relayout_data, trades_data):
    """Recompute markout averages for the visible time window.

    Loads its own data (no shared store) so this works as a standalone
    single-output callback — Dash 4's multi-output dispatch was unreliable.
    """
    if not event or strike is None:
        return ""

    snaps, fills = load_event_data(event)
    if snaps.empty:
        return ""

    init_fills = (
        fills[(fills["strike"] == strike)
              & (fills["client_order_id"].str.startswith("init_", na=False))]
        if not fills.empty and "client_order_id" in fills.columns
        else pd.DataFrame()
    )
    bbo_markouts = compute_markouts(init_fills, snaps)
    trade_markouts = {}
    if trades_data and "trades" in trades_data:
        trade_markouts = compute_trade_markouts(init_fills, trades_data["trades"])

    # Parse visible x range from relayout
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

    def filter_window(items):
        if not (x_min or x_max):
            return items
        out = []
        for ts_val, val_cents, _hover in items:
            ts = pd.Timestamp(ts_val)
            if x_min and ts < x_min:
                continue
            if x_max and ts > x_max:
                continue
            out.append((ts_val, val_cents, _hover))
        return out

    show_bbo = markout_mode in ("bbo", "both")
    show_trade = markout_mode in ("trade", "both")

    rows = []
    for sec, label in MARKOUT_INTERVALS:
        spans = [html.Span(f"{label}: ", style={"fontWeight": "bold",
                                                 "marginRight": "8px"})]
        added = False

        if show_bbo:
            window = filter_window(bbo_markouts.get(sec, []))
            if window:
                vals = [v for _, v, _ in window]
                avg = sum(vals) / len(vals)
                total = sum(vals)
                pos = sum(1 for v in vals if v >= 0)
                spans.append(html.Span(
                    f"BBO avg={avg:+.1f}c pnl={total:+.0f}c "
                    f"n={len(vals)} (+{pos}/-{len(vals)-pos})  ",
                    style={"color": "#facc15", "marginRight": "12px"},
                ))
                added = True

        if show_trade:
            window = filter_window(trade_markouts.get(sec, []))
            if window:
                vals = [v for _, v, _ in window]
                avg = sum(vals) / len(vals)
                total = sum(vals)
                pos = sum(1 for v in vals if v >= 0)
                spans.append(html.Span(
                    f"TRADE avg={avg:+.1f}c pnl={total:+.0f}c "
                    f"n={len(vals)} (+{pos}/-{len(vals)-pos})",
                    style={"color": "#06b6d4"},
                ))
                added = True

        if added:
            rows.append(html.Div(spans))
    return rows or ""


if __name__ == "__main__":
    app.run(debug=False, port=8051)
