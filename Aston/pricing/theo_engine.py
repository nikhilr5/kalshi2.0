"""Theo computation for a single 15-min up/down market.

Pure N(d2) — risk-neutral probability of finishing above the strike
at expiry under lognormal dynamics, treating the close as a single
point.  No TWAP-aware adjustment.
"""

import math


SECONDS_PER_YEAR = 365.25 * 24 * 3600


def prob_above(spot: float, strike: float, sigma: float,
               seconds_to_expiry: float, r: float = 0.0) -> float:
    """N(d2): risk-neutral probability that spot > strike at expiry.

    Args:
        spot: current spot price.
        strike: contract threshold (yes pays $1 if spot > strike).
        sigma: annualized vol (decimal, e.g. 0.6 for 60%).
        seconds_to_expiry: time remaining in seconds.
        r: continuous risk-free rate (decimal).  Default 0.

    Returns:
        Probability in [0, 1].  Edge cases (T<=0 or sigma<=0) return
        the deterministic 0/1 based on spot vs strike.
    """
    if seconds_to_expiry <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return 1.0 if spot > strike else 0.0
    T = seconds_to_expiry / SECONDS_PER_YEAR
    sqrt_T = math.sqrt(T)
    d2 = (math.log(spot / strike) + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    p = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    return max(0.0, min(1.0, p))


def compute_theo(spot: float, strike: float, sigma: float,
                 seconds_to_expiry: float) -> float | None:
    """Convenience wrapper — returns None when inputs aren't ready yet."""
    if spot <= 0 or strike <= 0 or sigma is None or sigma <= 0:
        return None
    if seconds_to_expiry <= 0:
        return None
    return prob_above(spot, strike, sigma, seconds_to_expiry)
