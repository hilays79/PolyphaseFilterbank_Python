#!/bin/bash
import numpy as np
from ipdb import set_trace as stop

def generate_sine_signal(n_taps, n_chan, n_windows, freq, include_noise=True):
    M = n_taps # Number of taps
    P = n_chan # Number of 'branches', also fft length
    W = n_windows # Number of windows of length M*P in input time stream
    freq = freq # Frequency of the input sine wave, in radians/sample (e.g. 0.1 for 0.1 cycles/sample)

    # Generate a test data stream
    samples = np.arange(M*P*W)
    if include_noise: noise = np.random.normal(loc=0.5, scale=0.1, size=M*P*W)
    else: noise = np.zeros(M*P*W)
    
    amp = 1
    cw_signal = amp * np.sin(samples * freq)
    data = noise + cw_signal
    return data

def generate_dirac_comb_signal(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=True):
    M = n_taps # Number of taps
    P = n_chan # Number of 'branches', also fft length
    W = n_windows # Number of windows of length M*P in input time stream
    delta_period = delta_period # Number of samples between each dirac pulse (e.g. 256 for a dirac comb with one pulse per P samples)
    delta_start = delta_start # Start position of the first dirac pulse


    # Generate a test data stream
    samples = np.arange(M*P*W)
    if include_noise: noise = np.random.normal(loc=0.5, scale=0.1, size=M*P*W)
    else: noise = np.zeros(M*P*W)

    # Create a dirac comb signal
    dirac_comb = np.zeros(M*P*W)
    dirac_comb[delta_start::delta_period] = 1
    data = noise + dirac_comb
    return data

if __name__ == "__main__":
    stop()