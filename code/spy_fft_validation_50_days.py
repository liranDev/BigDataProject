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
extrapolation = fourierExtrapolation(filtered_signal, n_predict)
pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label='extrapolation')
pl.plot(np.arange(0, x.size), detrended, 'b', label='x', linewidth=3)
pl.legend()
pl.show()

# validation
start = 0
end = 50 #time frame
result1 = []
result2 = []
result3 = []
result4 = []
result5 = []
result6 = []
result7 = []
idx = end-1
for day in range(len(filtered_signal)-end):
    extrapolation7 = fourierExtrapolation(filtered_signal[start:end], 7)
    if extrapolation7[idx+1] > extrapolation7[idx]: result1.append(extrapolation7[idx+1] - filtered_signal[idx])
    else: result1.append(filtered_signal[idx] - extrapolation7[idx+1])
    if extrapolation7[idx+2] > extrapolation7[idx]: result2.append(extrapolation7[idx+2] - filtered_signal[idx])
    else: result2.append(filtered_signal[idx] - extrapolation7[idx+2])
    if extrapolation7[idx+3] > extrapolation7[idx]: result3.append(extrapolation7[idx+3] - filtered_signal[idx])
    else: result3.append(filtered_signal[idx] - extrapolation7[idx+3])
    if extrapolation7[idx+4] > extrapolation7[idx]: result4.append(extrapolation7[idx+4] - filtered_signal[idx])
    else: result4.append(filtered_signal[idx] - extrapolation7[idx+4])
    if extrapolation7[idx+5] > extrapolation7[idx]: result5.append(extrapolation7[idx+5] - filtered_signal[idx])
    else: result5.append(filtered_signal[idx] - extrapolation7[idx+5])
    if extrapolation7[idx+6] > extrapolation7[idx]: result6.append(extrapolation7[idx+6] - filtered_signal[idx])
    else: result6.append(filtered_signal[idx] - extrapolation7[idx+6])
    if extrapolation7[idx+7] > extrapolation7[idx]: result7.append(extrapolation7[idx+7] - filtered_signal[idx])
    else: result7.append(filtered_signal[idx] - extrapolation7[idx+7])
    start += 1
    end += 1

print(pd.Series(result1).mean())
print(pd.Series(result2).mean())
print(pd.Series(result3).mean())
print(pd.Series(result4).mean())
print(pd.Series(result5).mean())
print(pd.Series(result6).mean())
expectation = [pd.Series(result1).mean(), pd.Series(result2).mean(), pd.Series(result3).mean(), pd.Series(result4).mean(),pd.Series(result5).mean(), pd.Series(result6).mean()]
plt.plot([1, 2, 3, 4, 5, 6], expectation)
plt.title('Profit Expectation per number of days look-ahead, 50 days timeframe')
plt.xlabel('Day')
plt.ylabel('Expectation')
plt.show()
plt.plot(result1)
plt.plot(result2)
plt.plot(result3)
plt.plot(result4)
plt.plot(result5)
plt.plot(result6)
plt.plot(result7)
plt.show()
