"""
Deribit Option-Implied Bracket Pricing

Uses Deribit's public (no auth) API to fetch BTC option prices, then
extracts the risk-neutral probability density via the Breeden-Litzenberger
butterfly method. This density is integrated over each Kalshi bracket
range to produce a theo price.

Why this is better than a vol smile:
    - Deribit options are liquid with tight spreads → clean input data
    - The butterfly method captures skew, kurtosis, and jump risk
      automatically — whatever the market prices in
    - No circular calibration: we price Kalshi brackets from an
      independent, liquid market (Deribit) rather than from Kalshi's
      own sparse orderbooks

How it works:
    1. Fetch all BTC call options for a target expiry from Deribit
    2. For each strike, get the mark price (in USD)
    3. Apply the Breeden-Litzenberger formula:
           density(K) ≈ (C(K-h) - 2*C(K) + C(K+h)) / h²
       This is the "butterfly" — the probability density at strike K
    4. For each Kalshi bracket [K_low, K_high], integrate (sum) the
       density over that range to get the bracket probability

Deribit details:
    - Public API, no auth needed
    - Options expire at 08:00 UTC
    - Instrument format: BTC-17APR26-72000-C
    - Prices returned in BTC, multiply by index_price for USD

Usage:
    pricer = DeribitBracketPricer()
    pricer.fetch_options("17APR26")          # load option chain
    theo = pricer.bracket_theo(70000, 70500) # price a bracket
    all_theos = pricer.all_bracket_theos(strikes)  # price all brackets
"""

import asyncio
import json
import time
import threading
import requests
import websockets
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from collections.abc import Callable


# =============================================================================
# Constants
# =============================================================================

# Deribit public API base URL (no auth needed)
_DERIBIT_BASE = "https://www.deribit.com/api/v2"

# Kalshi events expire in ET (Eastern Time)
# During EDT (March-November): ET = UTC-4
# During EST (November-March): ET = UTC-5
_ET_OFFSET_EDT = timedelta(hours=-4)
_ET_OFFSET_EST = timedelta(hours=-5)

# Map Kalshi series tickers to Deribit currency codes
# Deribit supports BTC and ETH options. SOL and DOGE are not available.
KALSHI_TO_DERIBIT_CURRENCY = {
    "KXBTC": "BTC",
    "KXBTCD": "BTC",
    "KXETH": "ETH",
    "KXETHD": "ETH",
}

# Deribit instrument name prefix per currency
# BTC options: BTC-17APR26-72000-C
# ETH options: ETH-17APR26-2000-C
DERIBIT_INSTRUMENT_PREFIX = {
    "BTC": "BTC",
    "ETH": "ETH",
}


# =============================================================================
# Deribit API helpers
# =============================================================================

def _deribit_get(method: str, params: dict = None) -> dict:
    """Call a Deribit public JSON-RPC method via GET.

    Deribit's REST API uses JSON-RPC over HTTP. Public methods don't
    need authentication.

    Args:
        method: e.g. "public/get_instruments"
        params: query parameters

    Returns:
        The "result" field from the JSON-RPC response.
    """
    url = f"{_DERIBIT_BASE}/{method}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data and data["error"]:
        raise RuntimeError(f"Deribit error: {data['error']}")
    return data.get("result", data)


def find_deribit_expiry(kalshi_close_time: str, currency: str = "BTC") -> str | None:
    """Find the closest Deribit expiry for a Kalshi close time.

    Kalshi close_time is an ISO string like "2026-04-17T21:00:00Z"
    (5pm ET = 21:00 UTC during EDT).

    Deribit options expire at 08:00 UTC. We first try an exact date
    match, then fall back to the nearest available Deribit expiry
    that expires ON or AFTER the Kalshi date.

    Returns Deribit expiry string like "17APR26", or None if no match.
    """
    if not kalshi_close_time:
        return None

    try:
        # Parse Kalshi close time (UTC)
        close_utc = datetime.fromisoformat(
            kalshi_close_time.replace("Z", "+00:00")
        )

        # Convert to ET to get the calendar date
        # Use EDT offset (UTC-4) for March-November
        month = close_utc.month
        if 3 <= month <= 11:
            et_time = close_utc + _ET_OFFSET_EDT
        else:
            et_time = close_utc + _ET_OFFSET_EST

        kalshi_date = et_time.date()

        # Format as Deribit expiry: "17APR26"
        day = et_time.day
        month_str = et_time.strftime("%b").upper()
        year_short = et_time.strftime("%y")
        exact_match = f"{day}{month_str}{year_short}"

        # Check if exact match exists on Deribit
        available = list_deribit_expiries(currency)
        if exact_match in available:
            return exact_match

        # No exact match — find the nearest expiry on or after Kalshi date
        # Parse each Deribit expiry string into a date for comparison
        _months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                   "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

        best = None
        best_date = None
        for exp_str in available:
            try:
                # Parse "17APR26" — day can be 1 or 2 digits
                for m_name, m_num in _months.items():
                    idx = exp_str.find(m_name)
                    if idx > 0:
                        d = int(exp_str[:idx])
                        y = 2000 + int(exp_str[idx + 3:])
                        exp_date = datetime(y, m_num, d).date()
                        # Only consider expiries on or after Kalshi date
                        if exp_date >= kalshi_date:
                            if best_date is None or exp_date < best_date:
                                best = exp_str
                                best_date = exp_date
                        break
            except Exception:
                continue

        if best:
            print(f"[Deribit] No exact match for {exact_match}, "
                  f"using nearest: {best}")
        return best

    except Exception:
        return None


def list_deribit_expiries(currency: str = "BTC") -> list[str]:
    """List all available option expiry dates on Deribit for a currency.

    Args:
        currency: "BTC" or "ETH"

    Returns list of expiry strings like ["17APR26", "18APR26", ...].
    Useful for finding what's available when there's no exact match.
    """
    instruments = _deribit_get("public/get_instruments", {
        "currency": currency,
        "kind": "option",
        "expired": "false",
    })

    # Extract unique expiry dates from instrument names
    # Format: BTC-17APR26-72000-C → "17APR26"
    expiries = set()
    for inst in instruments:
        name = inst.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) >= 3:
            expiries.add(parts[1])

    return sorted(expiries)


def find_weekly_deribit_expiry(currency: str = "BTC") -> str | None:
    """Find the nearest upcoming Friday Deribit expiry.

    Returns expiry string like "25APR26", or None if not found.
    """
    _months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
               "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    try:
        available = list_deribit_expiries(currency)
    except Exception:
        return None

    now = datetime.now(tz=timezone.utc).date()
    best = None
    best_date = None
    for exp_str in available:
        try:
            for m_name, m_num in _months.items():
                idx = exp_str.find(m_name)
                if idx > 0:
                    d = int(exp_str[:idx])
                    y = 2000 + int(exp_str[idx + 3:])
                    exp_date = datetime(y, m_num, d).date()
                    if exp_date.weekday() != 4:  # Friday only
                        break
                    if exp_date >= now:
                        if best_date is None or exp_date < best_date:
                            best = exp_str
                            best_date = exp_date
                    break
        except Exception:
            continue
    return best


# =============================================================================
# Core: Fetch option chain and build density
# =============================================================================

@dataclass
class OptionData:
    """A single option's market data."""
    strike: float
    call_price_usd: float  # mark price in USD
    bid_usd: float         # best bid in USD
    ask_usd: float         # best ask in USD
    mark_iv: float = 0.0   # mark implied volatility (annualised, e.g. 0.65 = 65%)
    bid_iv: float = 0.0    # bid implied volatility from Deribit
    ask_iv: float = 0.0    # ask implied volatility from Deribit
    open_interest: float = 0.0  # number of outstanding contracts


@dataclass
class DeribitBracketPricer:
    """Prices Kalshi brackets using Deribit option-implied density.

    The density is extracted via the Breeden-Litzenberger butterfly
    method from Deribit call option prices, with risk-free rate
    discounting:
        density(K) = e^(rT) * d²C/dK²

    Attributes:
        expiry_str:    Deribit expiry like "17APR26"
        index_price:   BTC spot/index price from Deribit
        risk_free_rate: annualised risk-free rate (default 0.0)
        options:       sorted list of OptionData (calls only, by strike)
        density:       list of (strike, prob_density) pairs
        _ready:        whether fetch_options() has been called successfully
    """

    currency: str = "BTC"            # Deribit currency: "BTC" or "ETH"
    expiry_str: str = ""
    index_price: float = 0.0
    risk_free_rate: float = 0.0   # annualised, e.g. 0.05 for 5%
    expiration_ts_ms: int = 0        # Deribit expiration timestamp (ms since epoch)
    options: list = field(default_factory=list)
    density: list = field(default_factory=list)
    _ready: bool = False

    def discover_instruments(self, expiry_str: str) -> list[str]:
        """Discover call option instrument names for an expiry via REST.

        This is a single REST call to get_instruments — fast. Returns
        the list of instrument names to subscribe to via WebSocket.
        """
        self.expiry_str = expiry_str
        self.expiration_ts_ms = 0
        self.options = []
        self.density = []
        self._ready = False

        all_instruments = _deribit_get("public/get_instruments", {
            "currency": self.currency,
            "kind": "option",
            "expired": "false",
        })

        call_names = []
        for inst in all_instruments:
            name = inst.get("instrument_name", "")
            parts = name.split("-")
            if (len(parts) == 4
                    and parts[1] == expiry_str
                    and parts[3] == "C"
                    and inst.get("is_active", False)):
                call_names.append(name)
                if self.expiration_ts_ms == 0:
                    self.expiration_ts_ms = inst.get("expiration_timestamp", 0) or 0

        print(f"[Deribit] Found {len(call_names)} calls for {expiry_str}")
        return sorted(call_names)

    def update_from_ticker(self, instrument_name: str, data: dict):
        """Update a single option from a WS ticker notification.

        Called by DeribitWsFeed for each incoming ticker update.
        """
        try:
            strike = float(instrument_name.split("-")[2])
        except (IndexError, ValueError):
            return

        idx_price = float(data.get("index_price", 0) or 0)
        if idx_price > 0:
            self.index_price = idx_price

        mark_btc = float(data.get("mark_price", 0) or 0)
        bid_btc = float(data.get("best_bid_price", 0) or 0)
        ask_btc = float(data.get("best_ask_price", 0) or 0)
        raw_mark_iv = data.get("mark_iv")
        mark_iv = float(raw_mark_iv) / 100.0 if raw_mark_iv else None
        raw_bid_iv = data.get("bid_iv")
        bid_iv = float(raw_bid_iv) / 100.0 if raw_bid_iv else None
        raw_ask_iv = data.get("ask_iv")
        ask_iv = float(raw_ask_iv) / 100.0 if raw_ask_iv else None
        oi = float(data.get("open_interest", 0) or 0)

        mark_usd = mark_btc * self.index_price
        bid_usd = bid_btc * self.index_price
        ask_usd = ask_btc * self.index_price

        # Update existing or insert new
        for opt in self.options:
            if opt.strike == strike:
                opt.call_price_usd = mark_usd
                opt.bid_usd = bid_usd
                opt.ask_usd = ask_usd
                if mark_iv is not None and mark_iv > 0:
                    opt.mark_iv = mark_iv
                if bid_iv is not None and bid_iv > 0:
                    opt.bid_iv = bid_iv
                if ask_iv is not None and ask_iv > 0:
                    opt.ask_iv = ask_iv
                if oi > 0:
                    opt.open_interest = oi
                return

        self.options.append(OptionData(
            strike=strike,
            call_price_usd=mark_usd,
            bid_usd=bid_usd,
            ask_usd=ask_usd,
            mark_iv=mark_iv if mark_iv is not None and mark_iv > 0 else 0.0,
            bid_iv=bid_iv if bid_iv is not None and bid_iv > 0 else 0.0,
            ask_iv=ask_iv if ask_iv is not None and ask_iv > 0 else 0.0,
            open_interest=oi,
        ))
        self.options.sort(key=lambda o: o.strike)

    def rebuild_density(self):
        """Rebuild probability density from current option prices.

        Call this after a batch of WS updates.
        """
        if len(self.options) < 5:
            return

        T = 0.0
        if self.expiration_ts_ms > 0:
            now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
            T = max((self.expiration_ts_ms - now_ms) / 1000.0 / (365.25 * 24 * 3600), 0.0)

        self._build_density(T)
        self._ready = True

    def fetch_options(self, expiry_str: str) -> bool:
        """Fetch the BTC call option chain from Deribit via REST (legacy).

        Kept for backwards compatibility. Prefer discover_instruments() +
        DeribitWsFeed for real-time data.
        """
        call_names = self.discover_instruments(expiry_str)
        if len(call_names) < 5:
            return False

        for name in call_names:
            try:
                ticker = _deribit_get("public/ticker", {
                    "instrument_name": name,
                })
                self.update_from_ticker(name, ticker)
            except Exception as e:
                print(f"[Deribit] Error fetching {name}: {e}")
                continue

        if len(self.options) < 5:
            print(f"[Deribit] Only {len(self.options)} usable options")
            return False

        print(f"[Deribit] Loaded {len(self.options)} call prices, "
              f"index={self.index_price:.0f}, "
              f"strikes=[{self.options[0].strike:.0f} ... {self.options[-1].strike:.0f}]")

        self.rebuild_density()
        return True

    def _build_density(self, T: float = 0.0):
        """Build the risk-neutral density from call prices.

        Uses the Breeden-Litzenberger butterfly formula for non-uniform
        strike spacing, with risk-free rate discounting:

            density(K_i) = e^(rT) * d²C/dK²

        where d²C/dK² is approximated as:
            (C(K_{i-1}) - 2*C(K_i) + C(K_{i+1})) / (h_left * h_right)

        The e^(rT) factor converts from the forward measure back to
        the spot measure. With r=0 this is just 1.0.

        After computing the raw density, we normalise so the total
        probability integrates to 1.0. This corrects for:
            - Truncated tails (strikes don't go to 0 or infinity)
            - Numerical noise from non-uniform strike spacing
        """
        import math

        opts = self.options
        n = len(opts)
        self.density = []

        # Discount factor: e^(rT)
        # With r=0 this is 1.0 (no effect). Setting r > 0 accounts
        # for the time value of money in the density extraction.
        discount_factor = math.exp(self.risk_free_rate * T)

        for i in range(1, n - 1):
            k_prev = opts[i - 1].strike
            k_curr = opts[i].strike
            k_next = opts[i + 1].strike

            c_prev = opts[i - 1].call_price_usd
            c_curr = opts[i].call_price_usd
            c_next = opts[i + 1].call_price_usd

            # Non-uniform step sizes
            h_left = k_curr - k_prev
            h_right = k_next - k_curr

            if h_left <= 0 or h_right <= 0:
                continue

            # Butterfly: second derivative for non-uniform spacing
            d2c = (c_prev / h_left
                   - c_curr * (1.0 / h_left + 1.0 / h_right)
                   + c_next / h_right) / ((h_left + h_right) / 2.0)

            # Apply discount factor: density = e^(rT) * d²C/dK²
            d2c *= discount_factor

            # Density must be non-negative (clamp noise)
            d2c = max(d2c, 0.0)

            self.density.append((k_curr, d2c))

        if self.density:
            # Compute raw total probability via trapezoidal integration
            raw_total = 0.0
            for i in range(1, len(self.density)):
                k_prev, d_prev = self.density[i - 1]
                k_curr, d_curr = self.density[i]
                dk = k_curr - k_prev
                raw_total += (d_prev + d_curr) / 2.0 * dk

            # Normalise density so total probability = 1.0
            # This accounts for truncated tails and numerical errors
            if raw_total > 0.01:
                scale = 1.0 / raw_total
                self.density = [(k, d * scale) for k, d in self.density]

            # Verify normalisation
            total = 0.0
            for i in range(1, len(self.density)):
                k_prev, d_prev = self.density[i - 1]
                k_curr, d_curr = self.density[i]
                dk = k_curr - k_prev
                total += (d_prev + d_curr) / 2.0 * dk

            print(f"[Deribit] Density built: {len(self.density)} points, "
                  f"raw_total={raw_total:.4f}, normalised={total:.4f}")

    def bracket_theo(self, k_low: float, k_high: float | None) -> float:
        """Price a single Kalshi bracket [k_low, k_high).

        Integrates the probability density over the bracket range
        using the trapezoidal rule.

        Args:
            k_low:  lower strike of the bracket
            k_high: upper strike (None for the top/unbounded bucket)

        Returns:
            Probability (0 to 1) that BTC expires in this range.
        """
        if not self._ready or not self.density:
            return 0.0

        # For the top bucket (no upper bound), use a large cap
        if k_high is None or k_high <= 0:
            k_high = self.density[-1][0] * 2.0

        # Integrate density over [k_low, k_high] using trapezoidal rule
        prob = 0.0
        for i in range(1, len(self.density)):
            k_a, d_a = self.density[i - 1]
            k_b, d_b = self.density[i]

            # Skip segments entirely outside the bracket
            if k_b <= k_low or k_a >= k_high:
                continue

            # Clip segment to bracket bounds
            seg_lo = max(k_a, k_low)
            seg_hi = min(k_b, k_high)

            if seg_hi <= seg_lo:
                continue

            # Interpolate density at clipped boundaries
            frac_lo = (seg_lo - k_a) / (k_b - k_a) if k_b > k_a else 0
            frac_hi = (seg_hi - k_a) / (k_b - k_a) if k_b > k_a else 0
            d_lo = d_a + frac_lo * (d_b - d_a)
            d_hi = d_a + frac_hi * (d_b - d_a)

            # Trapezoidal integration
            prob += (d_lo + d_hi) / 2.0 * (seg_hi - seg_lo)

        return max(prob, 0.0)

    def prob_above(self, K: float) -> float:
        """Probability that spot finishes above strike K.

        Uses the first derivative of call prices (Breeden-Litzenberger):
            P(S > K) = -e^(rT) * dC/dK

        This is more direct and less noisy than integrating the density
        (second derivative) since it only differences once.

        Falls back to density integration if K is outside the option
        strike range.
        """
        if not self._ready or not self.options:
            return 0.0

        import math

        # Find the two options straddling K
        opts = self.options
        for i in range(len(opts) - 1):
            if opts[i].strike <= K <= opts[i + 1].strike:
                h = opts[i + 1].strike - opts[i].strike
                if h <= 0:
                    continue
                dc_dk = (opts[i + 1].call_price_usd - opts[i].call_price_usd) / h
                # P(S > K) = -dC/dK (with discount factor)
                # For r=0 discount factor is 1.0
                T = 0.0  # approximate; exact T was used during density build
                discount = math.exp(self.risk_free_rate * T)
                prob = -dc_dk * discount
                return max(min(prob, 1.0), 0.0)

        # K is outside option range — fall back to density integration
        return self.bracket_theo(K, None)

    def prob_below(self, K: float) -> float:
        """Probability that spot finishes below strike K.

        Simply 1 - prob_above(K).
        """
        return 1.0 - self.prob_above(K)

    def _find_closest_iv(self, K: float) -> float:
        """Find the mark_iv (decimal) from the closest Deribit strike to K."""
        best_opt = None
        best_dist = float("inf")
        for opt in self.options:
            if opt.mark_iv > 0:
                dist = abs(opt.strike - K)
                if dist < best_dist:
                    best_dist = dist
                    best_opt = opt
        return best_opt.mark_iv if best_opt else 0.0

    def _find_closest_bid_iv(self, K: float) -> float:
        """Find the bid_iv (decimal) from the closest Deribit strike to K."""
        best_opt = None
        best_dist = float("inf")
        for opt in self.options:
            if opt.bid_iv > 0:
                dist = abs(opt.strike - K)
                if dist < best_dist:
                    best_dist = dist
                    best_opt = opt
        return best_opt.bid_iv if best_opt else 0.0

    def _find_closest_ask_iv(self, K: float) -> float:
        """Find the ask_iv (decimal) from the closest Deribit strike to K."""
        best_opt = None
        best_dist = float("inf")
        for opt in self.options:
            if opt.ask_iv > 0:
                dist = abs(opt.strike - K)
                if dist < best_dist:
                    best_dist = dist
                    best_opt = opt
        return best_opt.ask_iv if best_opt else 0.0

    def _find_closest_oi(self, K: float) -> float:
        """Find the open interest from the closest Deribit strike to K."""
        best_opt = None
        best_dist = float("inf")
        for opt in self.options:
            dist = abs(opt.strike - K)
            if dist < best_dist:
                best_dist = dist
                best_opt = opt
        return best_opt.open_interest if best_opt else 0.0

    @staticmethod
    def _bs_prob_above(S: float, K: float, sigma: float, T: float, r: float) -> float:
        """Black-Scholes P(S > K) = N(d2)."""
        import math
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        sqrt_T = math.sqrt(T)
        d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        return max(min(0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))), 1.0), 0.0)

    def prob_above_with_iv(self, K: float, sigma: float, spot: float = 0.0,
                           kalshi_close_iso: str = "") -> float:
        """Probability above K using Black-Scholes with an explicit IV (decimal).

        Args:
            K: strike price
            sigma: implied volatility (decimal, e.g. 0.65 for 65%)
            spot: live spot price. 0 = use Deribit index.
            kalshi_close_iso: Kalshi event close time ISO string. "" = use Deribit expiry.
        """
        if sigma <= 0:
            return 0.0
        S = spot if spot > 0 else self.index_price
        if S <= 0:
            return 0.0
        r = self.risk_free_rate
        now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
        if kalshi_close_iso:
            try:
                close_utc = datetime.fromisoformat(
                    kalshi_close_iso.replace("Z", "+00:00")
                )
                T = max((close_utc.timestamp() * 1000 - now_ms) / 1000.0 / (365.25 * 24 * 3600), 0.0)
            except Exception:
                T = 0.0
        elif self.expiration_ts_ms > 0:
            T = max((self.expiration_ts_ms - now_ms) / 1000.0 / (365.25 * 24 * 3600), 0.0)
        else:
            T = 0.0
        return self._bs_prob_above(S, K, sigma, T, r)

    def prob_above_bid_iv(self, K: float, spot: float = 0.0,
                          kalshi_close_iso: str = "") -> float:
        """Probability above K using closest strike's bid IV."""
        sigma = self._find_closest_bid_iv(K)
        return self.prob_above_with_iv(K, sigma, spot, kalshi_close_iso)

    def prob_above_ask_iv(self, K: float, spot: float = 0.0,
                          kalshi_close_iso: str = "") -> float:
        """Probability above K using closest strike's ask IV."""
        sigma = self._find_closest_ask_iv(K)
        return self.prob_above_with_iv(K, sigma, spot, kalshi_close_iso)

    def prob_above_iv(self, K: float, spot: float = 0.0,
                      kalshi_close_iso: str = "") -> float:
        """Probability above K using Black-Scholes with closest strike's IV.

        Uses the Kalshi spot price and Kalshi close time when provided,
        falling back to Deribit index price and Deribit expiry.

        Args:
            K: strike price
            spot: live spot price (Coinbase). 0 = use Deribit index.
            kalshi_close_iso: Kalshi event close time ISO string. "" = use Deribit expiry.
        """
        if not self.options:
            return 0.0

        sigma = self._find_closest_iv(K)
        if sigma <= 0:
            return 0.0

        S = spot if spot > 0 else self.index_price
        if S <= 0:
            return 0.0

        r = self.risk_free_rate

        # Time to expiry — use Kalshi close if provided, else Deribit expiry
        now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
        if kalshi_close_iso:
            try:
                close_utc = datetime.fromisoformat(
                    kalshi_close_iso.replace("Z", "+00:00")
                )
                T = max((close_utc.timestamp() * 1000 - now_ms) / 1000.0 / (365.25 * 24 * 3600), 0.0)
            except Exception:
                T = 0.0
        elif self.expiration_ts_ms > 0:
            T = max((self.expiration_ts_ms - now_ms) / 1000.0 / (365.25 * 24 * 3600), 0.0)
        else:
            T = 0.0

        return self._bs_prob_above(S, K, sigma, T, r)

    def all_bracket_theos(self, strikes: list[float]) -> list[float]:
        """Price all Kalshi brackets given sorted strike boundaries.

        Args:
            strikes: sorted list of bracket boundary strikes.
                     N strikes define N brackets:
                     [0, K1), [K1, K2), ..., [K_{n-1}, ∞)

        Returns:
            List of N theo probabilities, one per bracket.
        """
        theos = []
        for i in range(len(strikes)):
            k_low = strikes[i]
            k_high = strikes[i + 1] if i + 1 < len(strikes) else None
            theos.append(self.bracket_theo(k_low, k_high))
        return theos

    @property
    def ready(self) -> bool:
        """Whether the pricer has been successfully initialized."""
        return self._ready

    @property
    def n_options(self) -> int:
        """Number of options used in the density."""
        return len(self.options)

    @property
    def n_density_points(self) -> int:
        """Number of density points."""
        return len(self.density)

    @property
    def strike_range(self) -> tuple[float, float]:
        """(min_strike, max_strike) covered by the density."""
        if self.density:
            return (self.density[0][0], self.density[-1][0])
        return (0.0, 0.0)


# =============================================================================
# WebSocket Feed
# =============================================================================

_DERIBIT_WS_URL = "wss://www.deribit.com/ws/api/v2"


class DeribitWsFeed:
    """Streams live option prices from Deribit via WebSocket.

    Usage:
        feed = DeribitWsFeed(pricer, on_ready_callback)
        feed.start(["BTC-17APR26-72000-C", ...])
        # on_ready fires once all initial snapshots arrive
        # pricer is updated in real-time after that
        feed.stop()
    """

    def __init__(self, pricer: DeribitBracketPricer, on_update: Callable = None):
        self.pricer = pricer
        self.on_update = on_update  # called after each density rebuild
        self._instruments: list[str] = []
        self._received: set[str] = set()
        self._initial_done = False
        self._thread = None
        self._loop = None
        self._ws = None
        self._running = False
        self._rebuild_pending = False
        self._msg_id = 1
        self.last_update_ts: float = 0.0  # timestamp of last data received

    def start(self, instruments: list[str]):
        """Start WS connection on a background daemon thread."""
        self._instruments = list(instruments)
        self._received = set()
        self._initial_done = False
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._schedule_close)

    def _schedule_close(self):
        if self._ws:
            asyncio.ensure_future(self._safe_close())

    async def _safe_close(self):
        try:
            await self._ws.close()
        except Exception:
            pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        reconnect_delay = 2
        while self._running:
            try:
                self._loop.run_until_complete(self._connect_and_listen())
            except Exception as e:
                print(f"[DeribitWS] Loop error: {e}")
            if not self._running:
                break
            print(f"[DeribitWS] Disconnected, reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)  # backoff up to 30s
        try:
            self._loop.close()
        except Exception:
            pass

    async def _connect_and_listen(self):
        try:
            self._ws = await websockets.connect(
                _DERIBIT_WS_URL,
                ping_interval=20,
                ping_timeout=10,
            )
        except Exception as e:
            print(f"[DeribitWS] Connect error: {e}")
            return

        # Subscribe to ticker channels for all instruments
        channels = [f"ticker.{inst}.100ms" for inst in self._instruments]
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": "public/subscribe",
            "params": {"channels": channels},
        }
        self._msg_id += 1
        await self._ws.send(json.dumps(subscribe_msg))
        print(f"[DeribitWS] Subscribed to {len(channels)} ticker channels")

        # Start a periodic density rebuild task
        rebuild_task = asyncio.ensure_future(self._rebuild_loop())

        try:
            async for message in self._ws:
                if not self._running:
                    break
                self._handle_message(json.loads(message))
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[DeribitWS] Error: {e}")
        finally:
            rebuild_task.cancel()
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _rebuild_loop(self):
        """Rebuild density every 2 seconds if new data arrived.
        Also monitors for staleness — if no data for 30s, force reconnect."""
        _log_counter = 0
        _STALE_THRESHOLD = 30  # seconds with no data = stale
        while self._running:
            await asyncio.sleep(2)
            # Check for staleness — force reconnect if no data received
            if self.last_update_ts > 0 and (time.time() - self.last_update_ts) > _STALE_THRESHOLD:
                print(f"[DeribitWS] STALE — no data for {time.time() - self.last_update_ts:.0f}s, forcing reconnect")
                self.last_update_ts = 0.0  # reset so we don't spam
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                break  # exit rebuild loop → _connect_and_listen exits → outer loop reconnects
            if self._rebuild_pending:
                self._rebuild_pending = False
                self.pricer.rebuild_density()
                _log_counter += 1
                if _log_counter % 15 == 1:  # log every ~30s
                    ivs = [(o.strike, o.mark_iv * 100) for o in self.pricer.options
                           if o.mark_iv > 0]
                    if ivs:
                        sample = ivs[len(ivs)//2]  # mid-chain sample
                        print(f"[DeribitWS] IV sample: ${sample[0]:,.0f} = {sample[1]:.1f}%  "
                              f"({len(ivs)} opts with IV)")
                if self.on_update:
                    try:
                        self.on_update()
                    except Exception:
                        pass

    def _handle_message(self, data: dict):
        method = data.get("method")
        if method != "subscription":
            return

        params = data.get("params", {})
        channel = params.get("channel", "")
        ticker_data = params.get("data", {})

        if not channel.startswith("ticker."):
            return

        instrument = ticker_data.get("instrument_name", "")
        if not instrument:
            return

        self.pricer.update_from_ticker(instrument, ticker_data)
        self._rebuild_pending = True
        self.last_update_ts = time.time()

        # Track initial snapshot completion
        if not self._initial_done:
            self._received.add(instrument)
            if len(self._received) >= len(self._instruments):
                self._initial_done = True
                print(f"[DeribitWS] All {len(self._instruments)} initial "
                      f"snapshots received, building density")
                self.pricer.rebuild_density()
                if self.on_update:
                    try:
                        self.on_update()
                    except Exception:
                        pass
