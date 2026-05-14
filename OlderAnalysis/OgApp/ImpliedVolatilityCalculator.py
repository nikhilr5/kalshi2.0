from math import log, sqrt, exp
from scipy.stats import norm

# class to help calculate the implied volatility of a single option
class ImpliedVolatilityCalculator:
    def __init__(self):
        self.rate = 0.05
    
    #closed form solution
    # easy to solve for since we know N(d2) vs in B.S. we only know option price
    def solve_quadratic(self, underlying_price, market_price, strike_price, daysTilExpiration, option_type='call'):
        S = underlying_price
        K = strike_price
        T = daysTilExpiration / 365.0
        r = self.rate

        # clamp market price to valid range for norm.ppf
        market_price = max(0.01, min(0.99, market_price))

        # need meaningful time to expiry
        if T < 1e-6 or S <= 0 or K <= 0:
            return None

        if option_type == 'call':
            d2 = norm.ppf(market_price)
        else:
            d2 = -norm.ppf(market_price)

        a = 0.5 * T
        b = d2 * sqrt(T)
        c = -(log(S / K) + r * T)

        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None

        root1 = (-b + sqrt(discriminant)) / (2 * a)
        root2 = (-b - sqrt(discriminant)) / (2 * a)

        # take the smaller positive root
        positives = [v for v in [root1, root2] if v > 0]
        return min(positives) if positives else None
