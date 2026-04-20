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

import requests
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field


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
    "KXETH": "ETH",
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
    options: list = field(default_factory=list)
    density: list = field(default_factory=list)
    _ready: bool = False

    def fetch_options(self, expiry_str: str) -> bool:
        """Fetch the BTC call option chain from Deribit for a given expiry.

        Gets all call options, retrieves their mark prices, and builds
        the probability density using the butterfly method.

        Args:
            expiry_str: Deribit expiry format, e.g. "17APR26"

        Returns:
            True if successful (enough options to build density).
        """
        self.expiry_str = expiry_str
        self.options = []
        self.density = []
        self._ready = False

        # ---------------------------------------------------------------
        # Step 1: Get all option instruments for this expiry and currency
        # ---------------------------------------------------------------
        all_instruments = _deribit_get("public/get_instruments", {
            "currency": self.currency,
            "kind": "option",
            "expired": "false",
        })

        # Filter to calls for our expiry, and grab expiration timestamp
        call_names = []
        expiration_ts = None
        for inst in all_instruments:
            name = inst.get("instrument_name", "")
            parts = name.split("-")
            if (len(parts) == 4
                    and parts[1] == expiry_str
                    and parts[3] == "C"
                    and inst.get("is_active", False)):
                call_names.append(name)
                if expiration_ts is None:
                    # Deribit expiration_timestamp is in milliseconds
                    expiration_ts = inst.get("expiration_timestamp", 0)

        if len(call_names) < 5:
            print(f"[Deribit] Only {len(call_names)} calls found for {expiry_str}")
            return False

        print(f"[Deribit] Found {len(call_names)} calls for {expiry_str}")

        # ---------------------------------------------------------------
        # Step 2: Fetch mark prices for each call
        #
        # Deribit returns prices in BTC. Multiply by index_price for USD.
        # We use mark_price (Deribit's fair value) rather than last trade
        # because it's always available and less noisy.
        # ---------------------------------------------------------------
        options = []
        for name in sorted(call_names):
            try:
                ticker = _deribit_get("public/ticker", {
                    "instrument_name": name,
                })

                # Extract strike from instrument name: BTC-17APR26-72000-C
                strike = float(name.split("-")[2])

                # Index price (BTC spot) — same for all, grab from first
                idx_price = float(ticker.get("index_price", 0))
                if idx_price > 0:
                    self.index_price = idx_price

                # Prices in BTC → convert to USD
                mark_btc = float(ticker.get("mark_price", 0))
                bid_btc = float(ticker.get("best_bid_price", 0) or 0)
                ask_btc = float(ticker.get("best_ask_price", 0) or 0)

                mark_usd = mark_btc * self.index_price
                bid_usd = bid_btc * self.index_price
                ask_usd = ask_btc * self.index_price

                options.append(OptionData(
                    strike=strike,
                    call_price_usd=mark_usd,
                    bid_usd=bid_usd,
                    ask_usd=ask_usd,
                ))
            except Exception as e:
                print(f"[Deribit] Error fetching {name}: {e}")
                continue

        # Sort by strike
        options.sort(key=lambda o: o.strike)
        self.options = options

        if len(options) < 5:
            print(f"[Deribit] Only {len(options)} usable options")
            return False

        print(f"[Deribit] Loaded {len(options)} call prices, "
              f"index={self.index_price:.0f}, "
              f"strikes=[{options[0].strike:.0f} ... {options[-1].strike:.0f}]")

        # ---------------------------------------------------------------
        # Step 3: Compute time to expiry for discounting
        #
        # T is needed for the risk-free rate discount factor e^(rT).
        # Deribit expiration_timestamp is in milliseconds since epoch.
        # ---------------------------------------------------------------
        T = 0.0
        if expiration_ts and expiration_ts > 0:
            now_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
            T = max((expiration_ts - now_ms) / 1000.0 / (365.25 * 24 * 3600), 0.0)

        print(f"[Deribit] T={T:.6f} years, r={self.risk_free_rate:.4f}")

        # ---------------------------------------------------------------
        # Step 4: Build the probability density via butterfly method
        #
        # Breeden-Litzenberger formula with discounting:
        #   density(K) = e^(rT) * d²C/dK²
        #
        # We use the actual strike spacing (not uniform), so we compute
        # the second derivative using the three nearest strikes.
        # ---------------------------------------------------------------
        self._build_density(T)
        self._ready = True
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
