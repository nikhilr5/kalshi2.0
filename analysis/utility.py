import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
import zoneinfo

import numpy as np
import pandas as pd
import math
from statistics import NormalDist


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
                     show_spot: bool = True) -> None:
        """Launch interactive Dash app with buy/sell edge inputs for a single strike."""
        from dash import Dash, dcc, html, Input, Output
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        if iv_cols is None:
            iv_cols = ["smoothed_mid_iv"]

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
                html.Label("Strike: "),
                dcc.Dropdown(id="strike", options=[{"label": f"{k:.0f}", "value": k} for k in available_strikes],
                             value=strike, clearable=False, style={"width": "140px", "display": "inline-block", "verticalAlign": "middle"}),
                html.Label(" Buy Edge: ", style={"marginLeft": "20px"}), dcc.Input(id="buy-edge", type="number", value=0, step=0.01, style={"width": "80px"}),
                html.Label(" Sell Edge: ", style={"marginLeft": "20px"}), dcc.Input(id="sell-edge", type="number", value=0, step=0.01, style={"width": "80px"}),
                html.Label(" Smoothing Span: ", style={"marginLeft": "20px"}),
                dcc.Dropdown(id="span", options=[{"label": str(sp), "value": sp} for sp in available_spans],
                             value=60, clearable=False, style={"width": "100px", "display": "inline-block", "verticalAlign": "middle"}),
                html.Label(" IV Min: ", style={"marginLeft": "20px"}), dcc.Input(id="iv-min", type="number", value=0, step=1, style={"width": "60px"}),
                html.Label(" IV Max: ", style={"marginLeft": "10px"}), dcc.Input(id="iv-max", type="number", value=100, step=1, style={"width": "60px"}),
            ], style={"padding": "10px", "display": "flex", "alignItems": "center"}),
            dcc.Graph(id="strike-graph", style={"height": "90vh"}),
        ])

        @app.callback(Output("strike-graph", "figure"),
                       Input("strike", "value"), Input("buy-edge", "value"),
                       Input("sell-edge", "value"), Input("span", "value"),
                       Input("iv-min", "value"), Input("iv-max", "value"))
        def update_graph(selected_strike, buy_edge, sell_edge, span, iv_min, iv_max):
            buy_edge = buy_edge or 0
            sell_edge = sell_edge or 0
            span = span or 60
            selected_strike = selected_strike or strike
            theo_col = f"theo_fitted_{span}"
            iv_col = f"smoothed_mid_iv_{span}"

            s = df[df["strike"] == selected_strike].sort_values("ts").reset_index(drop=True)

            specs = [[{"secondary_y": True}], [{"secondary_y": True}]]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=[f"Strike {selected_strike:.0f} — Market vs Fitted Theo",
                                                f"Strike {selected_strike:.0f} — Smoothed IV over Time"],
                                specs=specs)

            if s.empty:
                return fig

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
                    buys = s[s["kalshi_yes_ask"] <= buy_theo]
                    if not buys.empty:
                        fig.add_trace(go.Scatter(x=buys["ts"], y=buys["kalshi_yes_ask"], name="Buy Signal",
                                                 mode="markers", marker=dict(color="green", size=8, symbol="triangle-up")),
                                      row=1, col=1)

                if sell_edge > 0:
                    fig.add_trace(go.Scatter(x=s["ts"], y=sell_theo, name="Sell Theo",
                                             line=dict(color="black", width=1, dash="dot"), opacity=0.4), row=1, col=1)
                    sells = s[s["kalshi_yes_bid"] >= sell_theo]
                    if not sells.empty:
                        fig.add_trace(go.Scatter(x=sells["ts"], y=sells["kalshi_yes_bid"], name="Sell Signal",
                                                 mode="markers", marker=dict(color="red", size=8, symbol="triangle-down")),
                                      row=1, col=1)

            if show_spot:
                fig.add_trace(go.Scatter(x=s["ts"], y=s["spot_mid"], name="BTC Spot",
                                         line=dict(color="green"), opacity=0.6),
                              row=1, col=1, secondary_y=True)

            if iv_col in s.columns:
                fig.add_trace(go.Scatter(x=s["ts"], y=s[iv_col] * 100, name=f"Smoothed IV (span={span})"), row=2, col=1)

            fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="BTC Price", row=1, col=1, secondary_y=True)
            iv_lo = iv_min if iv_min is not None else 0
            iv_hi = iv_max if iv_max is not None else 100
            fig.update_yaxes(title_text="Implied Vol (%)", range=[iv_lo, iv_hi], row=2, col=1, secondary_y=False)
            fig.update_xaxes(title_text="Time", row=2, col=1)

            fig.update_layout(height=800, template="plotly_white", hovermode="x unified",
                              uirevision="constant")
            return fig

        app.run(debug=False)