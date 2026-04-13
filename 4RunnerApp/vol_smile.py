"""
Volatility smile calibration from Kalshi bracket market prices.

Extracts implied volatilities from observed bracket prices, then fits a
3-parameter quadratic smile:

    sigma(k) = a + b * x + c * x^2

where x = ln(K / S) is the log-moneyness (strike relative to spot).

The three degrees of freedom are:
    a  — ATM (at-the-money) volatility level
    b  — skew: tilts the smile left/right (negative = put skew)
    c  — curvature (convexity): controls the "U" shape of the smile

Implied vol extraction:
    For each bracket with a valid mid-price, we numerically invert the
    Black-Scholes bracket probability using bisection. This approach
    works even when most brackets have no data (unlike the cumulative
    probability method which needs all brackets priced).

Usage:
    smile = VolSmile()
    smile.calibrate(spot, strikes, mid_prices, T)
    vol_for_bracket = smile.vol_at(strike)
    a, b, c = smile.params()
"""

import math
from dataclasses import dataclass, field


# =============================================================================
# Constants
# =============================================================================

_SQRT2 = math.sqrt(2.0)
_VOL_MIN = 0.01    # 1% annualised floor
_VOL_MAX = 10.0    # 1000% annualised ceiling
_BISECT_TOL = 1e-6  # bisection convergence tolerance
_BISECT_ITER = 80   # max bisection iterations


# =============================================================================
# Black-Scholes bracket probability
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _bracket_prob(spot: float, k_low: float, k_high: float | None,
                  T: float, sigma: float) -> float:
    """Probability that spot lands in [k_low, k_high] at expiry.

    Uses the Black-Scholes log-normal model with r = 0:
        P(K_low < S_T < K_high) = N(-d2(K_high)) - N(-d2(K_low))
        d2(K) = [ln(S/K) - sigma^2*T/2] / (sigma * sqrt(T))
    """
    if T <= 0 or sigma <= 0 or spot <= 0:
        return 0.0

    sqrt_t = sigma * math.sqrt(T)
    drift = -0.5 * sigma * sigma * T

    def p_below(k):
        d2 = (math.log(spot / k) + drift) / sqrt_t
        return _norm_cdf(-d2)

    p_high = p_below(k_high) if (k_high is not None and k_high > 0) else 1.0
    p_low = p_below(k_low) if (k_low is not None and k_low > 0) else 0.0
    return max(p_high - p_low, 0.0)


# =============================================================================
# Direct bracket implied vol via bisection
# =============================================================================

def _find_peak_vol(spot: float, k_low: float, k_high: float | None,
                   T: float) -> float:
    """Find the vol that maximises the bracket probability.

    The bracket probability is unimodal in sigma — it rises (as vol
    spreads probability into the bracket) then falls (as vol spreads it
    out further). Golden section search finds the peak.
    """
    golden = (math.sqrt(5) - 1) / 2
    a, b = _VOL_MIN, _VOL_MAX

    c = b - golden * (b - a)
    d = a + golden * (b - a)

    for _ in range(60):
        fc = _bracket_prob(spot, k_low, k_high, T, c)
        fd = _bracket_prob(spot, k_low, k_high, T, d)

        if fc > fd:
            b = d
        else:
            a = c

        c = b - golden * (b - a)
        d = a + golden * (b - a)

        if abs(b - a) < 1e-5:
            break

    return (a + b) / 2.0


def _implied_vol_bracket(spot: float, k_low: float, k_high: float | None,
                         T: float, market_price: float) -> float | None:
    """Extract implied vol from a bracket mid-price via bisection.

    Directly inverts P(K_low < S_T < K_high) = market_price for sigma.
    No cumulative probabilities needed — works bracket by bracket.

    The bracket probability is NOT monotonic in sigma (rises then falls),
    so we first find the peak, then bisect on the correct side.
    We prefer the left (lower vol) solution.

    Args:
        spot:         current spot price
        k_low:        lower strike of the bracket
        k_high:       upper strike (None for the top bucket)
        T:            time to expiry in years
        market_price: observed bracket mid-price (0 to 1)

    Returns:
        Implied annualised vol, or None if extraction fails.
    """
    if market_price <= 0.005 or market_price >= 0.995:
        return None
    if T <= 0 or spot <= 0 or k_low <= 0:
        return None

    # Find the peak — maximum probability achievable at any vol
    peak_vol = _find_peak_vol(spot, k_low, k_high, T)
    peak_prob = _bracket_prob(spot, k_low, k_high, T, peak_vol)

    if market_price > peak_prob:
        # Market price exceeds theoretical maximum — return peak vol
        return peak_vol

    # Bisect on the left side of the peak (lower vol, more meaningful)
    lo, hi = _VOL_MIN, peak_vol
    p_lo = _bracket_prob(spot, k_low, k_high, T, lo)

    # If target is below the minimum, try the right side instead
    if market_price < p_lo:
        lo, hi = peak_vol, _VOL_MAX

    for _ in range(_BISECT_ITER):
        mid = (lo + hi) / 2.0
        p_mid = _bracket_prob(spot, k_low, k_high, T, mid)

        if abs(p_mid - market_price) < _BISECT_TOL:
            return mid

        if mid <= peak_vol:
            # Left side: probability increases with vol
            if p_mid < market_price:
                lo = mid
            else:
                hi = mid
        else:
            # Right side: probability decreases with vol
            if p_mid > market_price:
                lo = mid
            else:
                hi = mid

    return (lo + hi) / 2.0


# =============================================================================
# Vol Smile — 3-parameter quadratic fit
# =============================================================================

@dataclass
class VolSmile:
    """Quadratic volatility smile: sigma(K) = a + b*x + c*x^2

    where x = ln(K / spot) is log-moneyness.

    Attributes:
        a:  ATM volatility level (intercept)
        b:  skew coefficient (tilt)
        c:  curvature coefficient (convexity / "smile")
    """

    # Fitted parameters
    a: float = 0.0   # ATM vol level
    b: float = 0.0   # skew
    c: float = 0.0   # curvature

    # Spot price used during calibration (needed for log-moneyness)
    _spot: float = 0.0

    # Calibration diagnostics
    _n_points: int = 0             # how many brackets had valid implied vols
    _residual: float = 0.0        # sum of squared residuals from the fit
    _calibrated: bool = False      # whether calibrate() has run successfully

    # Raw implied vols from last calibration: [(strike_mid, impl_vol), ...]
    _raw_ivs: list = field(default_factory=list)

    def calibrate(self, spot: float, strikes: list[float],
                  mid_prices: list[float], T: float) -> bool:
        """Calibrate the smile from bracket market data.

        For each bracket with a valid mid-price (> 0), numerically
        inverts the Black-Scholes bracket probability to find the
        implied vol via bisection. Then fits the 3-parameter quadratic
        smile via least squares.

        This works even when most brackets have no data — each bracket
        is solved independently (no cumulative probability needed).

        Args:
            spot:       current spot price
            strikes:    sorted list of bracket boundary strikes
                        (N strikes define N-1 brackets)
            mid_prices: mid-price for each bracket (len = len(strikes) - 1)
                        Use 0 for brackets with no market data
            T:          time to expiry in years

        Returns:
            True if calibration succeeded (>= 3 usable data points),
            False otherwise.
        """
        self._spot = spot
        self._raw_ivs = []
        self._calibrated = False

        if T <= 0 or spot <= 0:
            return False

        n_brackets = min(len(strikes) - 1, len(mid_prices))
        if n_brackets < 1:
            return False

        # -----------------------------------------------------------------
        # Step 1: Extract implied vol for each bracket independently
        #
        # For each bracket [strikes[i], strikes[i+1]) with a valid
        # mid-price, bisect to find the sigma that produces that price.
        # Assign the result to the bracket's midpoint strike.
        # -----------------------------------------------------------------
        for i in range(n_brackets):
            price = mid_prices[i]
            if price <= 0.005 or price >= 0.995:
                continue  # skip empty or extreme brackets

            k_low = strikes[i]
            k_high = strikes[i + 1] if i + 1 < len(strikes) else None

            if k_low <= 0:
                continue

            iv = _implied_vol_bracket(spot, k_low, k_high, T, price)
            if iv is not None and _VOL_MIN < iv < _VOL_MAX:
                # Use the bracket midpoint as the representative strike
                if k_high and k_high > 0:
                    strike_mid = (k_low + k_high) / 2.0
                else:
                    strike_mid = k_low
                self._raw_ivs.append((strike_mid, iv))

        self._n_points = len(self._raw_ivs)

        # -----------------------------------------------------------------
        # Step 2: Fit the 3-parameter quadratic smile
        #
        # sigma(K) = a + b*x + c*x^2  where x = ln(K/S)
        #
        # Solved via ordinary least squares (normal equations).
        # Need >= 3 points for 3 free parameters.
        # -----------------------------------------------------------------
        if self._n_points < 3:
            # Not enough data — use average vol if we have any points
            if self._n_points > 0:
                avg_vol = sum(iv for _, iv in self._raw_ivs) / self._n_points
                self.a = avg_vol
                self.b = 0.0
                self.c = 0.0
                self._calibrated = True
                return True
            return False

        # Build sums for the 3x3 normal equations:
        # | s0  s1  s2 | | a |   | sy   |
        # | s1  s2  s3 | | b | = | sxy  |
        # | s2  s3  s4 | | c |   | sx2y |
        s0 = s1 = s2 = s3 = s4 = 0.0
        sy = sxy = sx2y = 0.0

        for strike_mid, iv in self._raw_ivs:
            x = math.log(strike_mid / spot)
            x2 = x * x
            s0 += 1.0
            s1 += x
            s2 += x2
            s3 += x2 * x
            s4 += x2 * x2
            sy += iv
            sxy += x * iv
            sx2y += x2 * iv

        # Solve via Cramer's rule
        det = (s0 * (s2 * s4 - s3 * s3)
               - s1 * (s1 * s4 - s3 * s2)
               + s2 * (s1 * s3 - s2 * s2))

        if abs(det) < 1e-20:
            # Degenerate matrix — fall back to average vol
            avg_vol = sum(iv for _, iv in self._raw_ivs) / self._n_points
            self.a = avg_vol
            self.b = 0.0
            self.c = 0.0
            self._calibrated = True
            return True

        self.a = ((sy * (s2 * s4 - s3 * s3)
                    - s1 * (sxy * s4 - sx2y * s3)
                    + s2 * (sxy * s3 - sx2y * s2)) / det)

        self.b = ((s0 * (sxy * s4 - sx2y * s3)
                    - sy * (s1 * s4 - s3 * s2)
                    + s2 * (s1 * sx2y - s3 * sxy)) / det)

        self.c = ((s0 * (s2 * sx2y - s3 * sxy)
                    - s1 * (s1 * sx2y - s3 * sy)
                    + sy * (s1 * s3 - s2 * s2)) / det)

        # Sanity: ATM vol must be positive
        if self.a < _VOL_MIN:
            self.a = _VOL_MIN

        # Compute residual for diagnostics
        self._residual = 0.0
        for strike_mid, iv in self._raw_ivs:
            x = math.log(strike_mid / spot)
            fitted = self.a + self.b * x + self.c * x * x
            self._residual += (iv - fitted) ** 2

        self._calibrated = True
        return True

    def vol_at(self, strike: float) -> float:
        """Return the interpolated/extrapolated vol for a given strike.

        Uses the fitted quadratic: sigma(K) = a + b*x + c*x^2
        where x = ln(K / spot).

        Falls back to the ATM vol (a) if not calibrated or strike is invalid.
        """
        if not self._calibrated or self._spot <= 0 or strike <= 0:
            return self.a if self.a > 0 else 0.0

        x = math.log(strike / self._spot)
        vol = self.a + self.b * x + self.c * x * x

        # Clamp — don't let extrapolation go negative or insane
        return max(vol, _VOL_MIN)

    def params(self) -> tuple[float, float, float]:
        """Return the smile parameters (a, b, c).

        a = ATM vol level, b = skew, c = curvature.
        """
        return (self.a, self.b, self.c)

    @property
    def calibrated(self) -> bool:
        """Whether the smile has been successfully calibrated."""
        return self._calibrated

    @property
    def n_points(self) -> int:
        """Number of brackets used in the fit."""
        return self._n_points

    @property
    def raw_ivs(self) -> list[tuple[float, float]]:
        """Raw implied vols from last calibration: [(strike_mid, iv), ...]."""
        return list(self._raw_ivs)
