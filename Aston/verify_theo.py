"""Standalone theo verifier.

Plugs the inputs from a screenshot / live snapshot straight into
`compute_theo` (the function the app actually uses) and prints every
intermediate step so each piece can be eyeballed.

Edit the values under SNAPSHOT below.  No imports of app.py — only
theo_engine + math — so this is a clean reference.
"""

import math
import statistics
from theo_engine import compute_theo, prob_above, SECONDS_PER_YEAR


# -------- snapshot inputs (edit these) --------
SNAPSHOT = {
    "spot":   81050.59,    # $
    "strike": 81202.49,    # $
    "sigma":  0.230,       # annualized vol, decimal
    "secs":   297,         # seconds to close
}
# ----------------------------------------------


def main():
    s = SNAPSHOT
    spot, strike, sigma, secs = s["spot"], s["strike"], s["sigma"], s["secs"]

    print(f"{'spot':<22}= ${spot:,.2f}")
    print(f"{'strike':<22}= ${strike:,.2f}")
    print(f"{'sigma (annualized)':<22}= {sigma*100:.2f}%")
    print(f"{'seconds to close':<22}= {secs}s  ({secs/60:.2f} min)")
    print()

    # Step 1: convert T to years
    T = secs / SECONDS_PER_YEAR
    print(f"T (years)             = {T:.10g}")

    # Step 2: σ²T (variance) and σ√T (std dev in log space)
    var_T = sigma * sigma * T
    std_logS = math.sqrt(var_T)
    print(f"σ²·T                  = {var_T:.10g}")
    print(f"σ·√T (1σ log-move)    = {std_logS:.10g}")
    print(f"1σ price move         = ${spot * std_logS:,.2f} "
          f"({std_logS*100:.4f}% of spot)")
    print()

    # Step 3: log-moneyness and drift correction
    log_sk = math.log(spot / strike)
    drift = -0.5 * sigma * sigma * T  # r = 0 assumed by compute_theo
    print(f"ln(S/K)               = {log_sk:.10g}")
    print(f"drift  (-0.5σ²T)      = {drift:.10g}")
    print(f"  numerator           = {log_sk + drift:.10g}")
    print()

    # Step 4: d2
    d2 = (log_sk + drift) / std_logS
    print(f"d2                    = {d2:+.6f}")
    print(f"  → strike is {-d2:.2f}σ from spot in log space")
    print()

    # Step 5: N(d2) via erf (matches theo_engine internals)
    p_erf = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

    # Cross-check via statistics.NormalDist
    p_norm = statistics.NormalDist().cdf(d2)

    print(f"N(d2) via erf         = {p_erf:.8f}")
    print(f"N(d2) via NormalDist  = {p_norm:.8f}   (cross-check)")
    print(f"|erf − norm|          = {abs(p_erf - p_norm):.2e}")
    print()

    # Step 6: route through the actual app entry point
    theo_app = compute_theo(spot, strike, sigma, secs)
    theo_pa  = prob_above(spot, strike, sigma, secs)
    print(f"compute_theo(...)     = {theo_app:.8f}  ({theo_app*100:.2f}¢)")
    print(f"prob_above(...)       = {theo_pa:.8f}  ({theo_pa*100:.2f}¢)")
    print()

    # Step 7: implied σ given the displayed theo, for round-trip check
    # Solve N(d2) = theo for σ via bisection.
    target = theo_app
    def f(sig):
        return prob_above(spot, strike, sig, secs) - target
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            # As σ increases, OTM theo increases — sign convention.
            lo = mid
        else:
            hi = mid
    σ_back = 0.5 * (lo + hi)
    print(f"σ implied from theo   = {σ_back*100:.4f}% "
          f"(should match input {sigma*100:.4f}%)")


if __name__ == "__main__":
    main()
