import pandas as pd
from CONFIG import *
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy import signal
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

#plt.figure(figsize = (8, 6))
#plt.plot(t, x, 'r')
#plt.ylabel('Amplitude')
#plt.show()

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

threshold = 500
plt.plot(np.arange(0, threshold, 1), ifft(X[0:threshold]), 'r')
plt.title('Clean Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

plt.plot(np.arange(threshold, 4279, 1), ifft(X[threshold:4279]), 'r')
plt.title('Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

#detrended = signal.detrend(close_prices)
#trend = close_prices - detrended
#plt.plot(detrended)
#plt.show()
#plt.plot(trend)
#plt.show()