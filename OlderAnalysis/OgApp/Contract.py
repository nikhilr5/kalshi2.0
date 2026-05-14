from datetime import datetime

#domain object to hold onto contract state
class Contract:
    def __init__(self, type:str, strike: int, expiration: datetime):
        self.type = type
        self.strike = strike
        self.expiration = expiration
        self.bestBid = 0
        self.bestOffer = 0
        self.implied_vol = 0
    
    def get_midpoint(self):
        return (self.bestBid + self.bestOffer) / 2
    
    def getDaysTilExpiration(self) -> float:
        delta = self.expiration - datetime.now()
        return delta.total_seconds() / 86400