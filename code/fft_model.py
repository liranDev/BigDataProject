import pandas as pd
from config import *
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import lfilter
import pylab as pl

def fourierExtrapolation(x, n_predict):
    n = x.size
    #n_harm = 10  # number of harmonics in model
    n_harm = x.size  # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)  # find linear trend in x
    #x_notrend = x - p[0] * t  # detrended x
    x_freqdom = fft(x)  # detrended x in frequency domain
    f = fftfreq(n)  # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

def predict(data):
    detrended = signal.detrend(data)
    n = 15
    b = [1.0 / n] * n
    a = 1
    filtered_signal = lfilter(b, a, detrended)
    extrapolation = fourierExtrapolation(filtered_signal, 3)
    return extrapolation