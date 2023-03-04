import pandas as pd
from config import *
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lfilter
import pylab as pl
import multiprocessing
from multiprocessing import Pool
import pickle
import fft_model
import sentiment_model
import ensemble_model
cpus = multiprocessing.cpu_count()
spy_df = pd.read_csv(SPY_STREAM)
close_prices = spy_df['Close'].to_numpy()
sentiment_df = pd.read_csv(DATA_STREAM)
sentences = sentiment_df['Sentence']
#parameters:
start = 0
window = 50 #time frame

if __name__ == '__main__':
    for batch in range(0, round(len(close_prices))):
        decision = ensemble_model.analyze(close_prices[start:start+window], sentiment_df[start:start+window])
        start += 1
