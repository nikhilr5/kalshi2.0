import numpy as np
import matplotlib.pyplot as plt

class VolatilitySmile:
    def __init__(self):
        pass

    # given a set of strikes and implied_vol update the coeffs for the regression curve
    def fit(self, strikes, implied_vol):
        coeffs = np.polyfit(strikes, implied_vol, 2)
        self.b2 = coeffs[0] #x^2
        self.b1 = coeffs[1] #x
        self.b0 = coeffs[2] #y-intercept
        return coeffs
        
    # given a strike solve for the implied vol
    def solve(self, strike):
        solved = self.b0 + self.b1 * strike + self.b2 * (strike **2)
        return solved

    #strike_range = low and high of the ranges of strikes you want to plot
    def graph_smile(self, strike_range):
        x = np.linspace(strike_range[0], strike_range[1], 100)
        y = self.b0 + self.b1 * x + self.b2 * x**2

        plt.plot(x, y, color='blue')
        plt.xlabel('Strike')
        plt.ylabel('Implied Vol')
        plt.title('Volatility Smile')
        plt.savefig('vol_smile.png')
        plt.show()

    # add a check to make sure the smile was updated recently before using
    def check_last_update():
        pass