import pandas as pd
import numpy as np
from config import *
import ensemble_model
import portfolio_management_model

spy_df = pd.read_csv(SPY_STREAM)
close_prices = spy_df['Close'].to_numpy()
sentiment_df = pd.read_csv(DATA_STREAM)
sentences = sentiment_df['Sentence']
#parameters:
start = 0
window = 50 #time frame
results = []
if __name__ == '__main__':
    for batch in range(0, round(len(close_prices))):
        decision = ensemble_model.analyze(close_prices[start:start+window], sentiment_df[start:start+window])
        portfolio_management_model.implement(decision)
        result = portfolio_management_model.get_results(decision, close_prices[start+window+1])
        results.append(result)
        ensemble_model.optimize(results, decision[1], decision[2])
        start += 1
