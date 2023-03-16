import pandas as pd
import math




def mean_sentiment(lst):
    return pd.Series(lst).mean()
def mean_rounded(lst):
    return round(pd.Series(lst).mean())
def exponent(lst):
    new = [lst[i]*math.exp(abs(lst[i])*i) for i in range(1, len(lst))]
    scaled = pd.Series(new)/(pd.Series(new).max())
    #scaled.plot(kind='bar')
    return scaled.mean()
