"""HAR-RV runtime estimator on Parkinson (high-low) RV.

Each 1-minute "bucket" stores the high and low seen in that minute.
Per-minute variance = ln(H/L)² / (4·ln 2) — the Parkinson estimator,
which uses the range of the minute rather than just the close.  More
efficient than close-to-close and meaningfully less biased on BTC/ETH
1-min candles (~3-7% vs ~+7% for cc_1min — see rv_compare.py).

Tick aggregation: on_price() tracks the running max/min of the
current minute and commits a (minute, high, low) tuple when the minute
boundary crosses.

Coefficients are loaded from har_coefficients.json (produced by
har_fit.py).  The fitter MUST be run against this same Parkinson
predictor — close-to-close coefficients aren't valid for Parkinson
inputs.  If the file is missing a sane prior is used.
"""

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Iterable

ANN_MINUTES = 365.25 * 24 * 60
H_15, H_30, H_4H, H_24H = 15, 30, 240, 1440
FOUR_LN2 = 4.0 * math.log(2.0)

# Used when no fitted coefficients are on disk.  Same shape used for
# cc_1min HAR; will be replaced by the next har_fit.py run.
PRIOR = {
    "beta0":   0.0,
    "beta_15": 0.40,
    "beta_30": 0.25,
    "beta_4h": 0.20,
    "beta_24h": 0.15,
}


def _annualize(sq_sum: float, window_minutes: int) -> float:
    if sq_sum <= 0:
        return 0.0
    return math.sqrt(sq_sum * (ANN_MINUTES / window_minutes))


def _parkinson_var(high: float, low: float) -> float:
    """Per-minute variance estimate from one minute's range.
    Returns 0 for degenerate inputs (zero range, bad data)."""
    if high <= 0 or low <= 0 or high <= low:
        return 0.0
    return math.log(high / low) ** 2 / FOUR_LN2


class HARRVEstimator:

    def __init__(self, coef_path: Path | None = None):
        # 1,440 minute-buckets cover the longest horizon (24h).
        # Each entry: (minute_idx, high, low).
        self._buckets: deque = deque(maxlen=H_24H)

        # In-progress current minute — running max/min until the next
        # minute boundary commits it.
        self._curr_minute: int | None = None
        self._curr_high: float = 0.0
        self._curr_low: float = float("inf")

        self.rv_15m: float | None = None
        self.rv_30m: float | None = None
        self.rv_4h: float | None = None
        self.rv_24h: float | None = None
        self.forecast: float | None = None

        self.coef = dict(PRIOR)
        self.coef_source = "prior"
        self.r2_train: float | None = None
        self.r2_test:  float | None = None
        self.fit_at:   str | None = None
        self.n_train:  int | None = None
        self.estimator_label = "parkinson"
        if coef_path and coef_path.exists():
            self._load_coefficients(coef_path)

    def _load_coefficients(self, path: Path):
        try:
            with path.open() as f:
                d = json.load(f)
            for k in ("beta0", "beta_15", "beta_30", "beta_4h", "beta_24h"):
                self.coef[k] = float(d[k])
            self.r2_train = d.get("r2_train")
            self.r2_test  = d.get("r2_test")
            self.fit_at   = d.get("fit_at")
            self.n_train  = d.get("n_train")
            # Refuse silently-wrong cc-trained coefficients.
            saved_est = d.get("estimator", "cc_1min")
            if saved_est != self.estimator_label:
                print(f"[HAR-RV] coefficient file was fit with "
                      f"estimator={saved_est!r} but runtime expects "
                      f"{self.estimator_label!r}; using prior instead")
                self.coef = dict(PRIOR)
                self.coef_source = "prior (estimator mismatch)"
            else:
                self.coef_source = "fitted"
        except Exception as e:
            print(f"[HAR-RV] failed to load {path}: {e}; using prior")

    def seed_from_candles(self, candles: Iterable[tuple[int, float, float]]):
        """Seed the buffer from historical 1-minute (high, low) bars.

        Args:
            candles: iterable of (unix_minute_index, high, low).
                unix_minute_index = unix_seconds // 60.  Order
                doesn't matter — we sort and de-dupe.
        """
        rows = sorted({(int(t), float(h), float(l)) for t, h, l in candles
                       if h > 0 and l > 0})
        if not rows:
            return
        existing = {b[0] for b in self._buckets}
        for t, h, l in rows:
            if t not in existing:
                self._buckets.append((t, h, l))
        # Re-sort + trim (deque maxlen handles trim, but we may have
        # inserted out-of-order).
        merged = sorted(self._buckets, key=lambda b: b[0])[-H_24H:]
        self._buckets.clear()
        for entry in merged:
            self._buckets.append(entry)
        if self._curr_minute is None and self._buckets:
            last_min, last_h, last_l = self._buckets[-1]
            self._curr_minute = last_min
            self._curr_high = last_h
            self._curr_low = last_l
        self._recompute()

    def on_price(self, price: float, ts: float | None = None):
        """Feed one Coinbase tick.  Tracks the running high/low of the
        current minute; commits one (minute, high, low) tuple per
        minute boundary, then refreshes per-horizon RV and forecast."""
        if price <= 0:
            return
        t = ts if ts is not None else time.time()
        minute = int(t // 60)

        if self._curr_minute is None:
            self._curr_minute = minute
            self._curr_high = price
            self._curr_low = price
            return

        if minute > self._curr_minute:
            self._buckets.append(
                (self._curr_minute, self._curr_high, self._curr_low))
            self._curr_minute = minute
            self._curr_high = price
            self._curr_low = price
            self._recompute()
            return

        # Same minute — update running extremes.
        if price > self._curr_high:
            self._curr_high = price
        if price < self._curr_low:
            self._curr_low = price

    def _recompute(self):
        n = len(self._buckets)
        if n < H_24H:
            return
        # Per-minute Parkinson variance.  Order matches the deque.
        v = [_parkinson_var(b[1], b[2]) for b in self._buckets]

        self.rv_15m = _annualize(sum(v[-H_15:]),  H_15)
        self.rv_30m = _annualize(sum(v[-H_30:]),  H_30)
        self.rv_4h  = _annualize(sum(v[-H_4H:]),  H_4H)
        self.rv_24h = _annualize(sum(v[-H_24H:]), H_24H)

        f = (self.coef["beta0"]
             + self.coef["beta_15"] * self.rv_15m
             + self.coef["beta_30"] * self.rv_30m
             + self.coef["beta_4h"] * self.rv_4h
             + self.coef["beta_24h"] * self.rv_24h)
        self.forecast = max(0.0, f)

    # --- Surface compatibility with RealizedVolEstimator + callers. ---
    def get_annualized_vol(self) -> float | None:
        return self.forecast

    def sample_count(self) -> int:
        return len(self._buckets)

    def horizon_breakdown(self) -> dict:
        return {
            "rv_15m":      self.rv_15m,
            "rv_30m":      self.rv_30m,
            "rv_4h":       self.rv_4h,
            "rv_24h":      self.rv_24h,
            "forecast":    self.forecast,
            "coef":        dict(self.coef),
            "coef_source": self.coef_source,
            "r2_train":    self.r2_train,
            "r2_test":     self.r2_test,
            "fit_at":      self.fit_at,
            "n_train":     self.n_train,
            "samples":     len(self._buckets),
            "estimator":   self.estimator_label,
        }
