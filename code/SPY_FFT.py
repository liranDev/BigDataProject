import pandas as pd
from CONFIG import *
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

data_df = pd.read_csv(SPY_STREAM)

close_prices = data_df['Close'].to_numpy()
plt.plot(close_prices)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

detrended = signal.detrend(close_prices)
trend = close_prices - detrended
# sampling rate
sr = 4279
# sampling interval
ts = 1.0/sr
t = np.arange(0, 1, ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7
x += 0.5* np.sin(2*np.pi*freq*t)

#X = fft(close_prices)
X = fft(detrended)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

#plt.figure(figsize = (8, 6))
#plt.plot(np.arange(0, 4279, 1), close_prices, 'r')
#plt.ylabel('Amplitude')
#plt.show()

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.title('Detrended Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

n = 15
b = [1.0/n]*n
a = 1
filtered_signal = lfilter(b, a, detrended)
plt.plot(np.arange(0, 4279, 1), filtered_signal, 'r')
plt.title('Clean Signal using lfilter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

noise = detrended - filtered_signal
plt.plot(np.arange(0, 4279, 1), noise, 'r')
plt.title('Noise using lfilter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

#threshold = 500
#plt.plot(np.arange(0, threshold, 1), ifft(X[0:threshold]), 'r')
#plt.title('Clean Signal')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.tight_layout()
#plt.show()

#plt.plot(np.arange(threshold, 4279, 1), ifft(X[threshold:4279]), 'r')
#plt.title('Noise')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.tight_layout()
#plt.show()

n_predict = 3
extrapolation = fourierExtrapolation(detrended, n_predict)
pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label='extrapolation')
pl.plot(np.arange(0, x.size), detrended, 'b', label='x', linewidth=3)
pl.legend()
pl.show()
