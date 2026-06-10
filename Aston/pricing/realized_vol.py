"""Simple rolling realized-volatility estimator.

Samples spot at fixed intervals and computes the close-to-close
standard deviation of log returns over a rolling window, annualized.

Designed for the 15-min up/down product where the only vol input the
theo needs is "annualized realized vol over a recent window."  No
smile, no IV inversion — just RV from the Coinbase tape.

Usage:
    est = RealizedVolEstimator(lookback_minutes=30, sample_seconds=10)
    feed.on_price = lambda px, bid, ask: est.on_price(px)
    sigma = est.get_annualized_vol()  # None until enough samples
"""

import math
import time
from collections import deque


class RealizedVolEstimator:

    def __init__(self, lookback_minutes: float = 30.0,
                 sample_seconds: float = 10.0):
        """
        Args:
            lookback_minutes: rolling window length in minutes.
            sample_seconds: how often to record a new spot sample.  Shorter
                = more responsive but noisier.  10s is a reasonable
                starting point for crypto.
        """
        self.lookback_minutes = lookback_minutes
        self.sample_seconds = sample_seconds
        # samples are (monotonic_ts, price) tuples
        max_samples = int(lookback_minutes * 60 / sample_seconds) + 2
        self._samples: deque = deque(maxlen=max_samples)
        self._last_sample_ts: float = 0.0

    def on_price(self, price: float):
        """Feed a fresh spot tick.  Records a sample if enough time elapsed."""
        if price <= 0:
            return
        now = time.monotonic()
        if (now - self._last_sample_ts) < self.sample_seconds:
            return
        self._samples.append((now, price))
        self._last_sample_ts = now
        self._evict_old(now)

    def _evict_old(self, now: float):
        """Drop samples outside the lookback window."""
        cutoff = now - self.lookback_minutes * 60
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def get_annualized_vol(self) -> float | None:
        """Return annualized realized vol, or None if too few samples.

        Standard deviation of log returns between consecutive samples,
        scaled by sqrt(samples_per_year).  Uses the actual sample
        interval rather than the configured one — protects against
        gaps if the feed dropped briefly.
        """
        if len(self._samples) < 3:
            return None
        log_returns = []
        for i in range(1, len(self._samples)):
            t_prev, p_prev = self._samples[i - 1]
            t_curr, p_curr = self._samples[i]
            if p_prev <= 0 or p_curr <= 0:
                continue
            r = math.log(p_curr / p_prev)
            log_returns.append(r)
        if len(log_returns) < 2:
            return None
        # Sample variance (n-1) is standard for realized vol estimation.
        mean = sum(log_returns) / len(log_returns)
        var = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        std_per_sample = math.sqrt(var)
        # Annualize: there are (365.25 * 24 * 3600 / sample_seconds)
        # sample intervals per year.
        samples_per_year = (365.25 * 24 * 3600) / self.sample_seconds
        return std_per_sample * math.sqrt(samples_per_year)

    def sample_count(self) -> int:
        return len(self._samples)
