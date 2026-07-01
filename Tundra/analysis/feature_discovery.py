#look into which asos features most impact the market 
#cloud cover, humaity, etc


#first look at big market changes and see what feature changes could have caused it
#then go the other way and see if those 

from util import market_swings, swings_with_features
sw = market_swings("2026-06-01", "2026-06-22") 

swings_features = swings_with_features(sw, asos='NYC', lookback_hours=3).drop(columns=['event_day', 'peak', 'cloud_at'])
print(swings_features[swings_features['cloud_jump'] == True])
print(swings_features.columns)
print(len(swings_features))