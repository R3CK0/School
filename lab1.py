from numpy.fft import fft, ifft, fftshift, fftfreq
import numpy as np
import pylab
import matplotlib.pyplot as plt

#function x
def x1(n):
    return np.sin(0.1*pylab.pi*n+pylab.pi/4)

def x2(n):
    return (-1)**n

def x3(n):
    return (n == 10).astype(float)

def window(n):
    return np.hanning(n)

def plotfft(signal, n, window, norm_freq=False):
    freq = fftshift(fftfreq(n))
    if norm_freq:
        freq *= 2*np.pi
    fft_input = signal(np.arange(n))*window(n)
    spectrum = fftshift(fft(fft_input))
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, np.abs(spectrum))
    if norm_freq:
        plt.xlabel('omega')
    else:
        plt.xlabel('Freq')
    plt.ylabel('Amplitude')
    plt.subplot(122)
    plt.stem(freq, np.angle(spectrum))
    if norm_freq:
        plt.xlabel('omega')
    else:
        plt.xlabel('Freq')
    plt.ylabel('phase')
    plt.show()

N = 12

plotfft(x1, N, window)
plotfft(x2, N, window)
plotfft(x3, N, window)


