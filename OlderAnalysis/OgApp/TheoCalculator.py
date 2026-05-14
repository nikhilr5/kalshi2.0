from Contract import Contract
from scipy.stats import norm
import math


class TheoCalculator:
    def __init__(self):
        self.rate = 0.05
        pass

    #pass in vol estimate and return theo for binary option
    def calculate(self, underlying, vol_estimate, contract: Contract):
        T = contract.getDaysTilExpiration() / 365
        d2 = (math.log(underlying / contract.strike) + (self.rate - (vol_estimate**2) / 2) * T) / (vol_estimate * math.sqrt(T))        
        proba = norm.cdf(d2)
        return proba