import pandas as pd
import numpy as np
from config import *
import ensemble_model
import portfolio_management_model
import matplotlib.pyplot as plt
from sklearn import metrics

spy_df = pd.read_csv(SPY_STREAM)
close_prices = spy_df['Close'].to_numpy()
sentiment_df = pd.read_csv(DATA_STREAM)
sentences = sentiment_df['Sentence']
#parameters:
start = 0
window = 50 #time frame
results = []
fft_factors = []
sent_factors = []
decisions = []
outcomes = []
#if __name__ == '__main__':
# for batch in range(0, round(len(close_prices))-1):
for batch in range(0, 285):
    decision = ensemble_model.analyze(close_prices[start:start+window], sentiment_df[start:start+window], fft_factors, sent_factors, outcomes)
    decisions.append(decision[0])
    fft_factors.append(decision[3])
    sent_factors.append(decision[4])
    portfolio_management_model.implement(decision)
    result = portfolio_management_model.get_results(decision, close_prices[start+window+1])
    results.append(result)
    outcome = portfolio_management_model.get_outcome(close_prices[start + window], close_prices[start + window + 1])
    outcomes.append(outcome)
    ensemble_model.optimize(results, decision[1], decision[2])
    print(f'{start} out of {len(close_prices-1)}')
    start += 1

print('test')
# plt.plot(results)
# plt.show()

print("Accuracy:", metrics.accuracy_score(outcomes[100:], decisions[100:]))

acc = []
for i in range(0, 185):
    acc.append(metrics.accuracy_score(outcomes[i:i+100], decisions[i:i+100]))
plt.plot(acc)
plt.show()
