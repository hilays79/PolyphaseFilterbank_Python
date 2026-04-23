#!/bin/bash
import numpy as np
from ipdb import set_trace as stop
import numpy as np

def generate_sine_signal(n_taps, n_chan, n_windows, freq, include_noise=True, complex_sine=False):
    M = n_taps     # Number of taps
    P = n_chan     # Number of 'branches', also fft length
    W = n_windows  # Number of windows of length M*P in input time stream
    
    freq_arr = np.atleast_1d(freq)[:, np.newaxis]
    num_freqs = freq_arr.shape[0]
    total_samples = M * P * W

    samples = np.arange(total_samples)
    
    # 1. Generate the CW signal
    amp = 1
    if complex_sine:
        cw_signal = amp * np.exp(1j * freq_arr * samples)
    else:
        cw_signal = amp * np.sin(freq_arr * samples)
    
    # 2. Generate the noise
    if include_noise: 
        # Generate real noise and cast it to complex immediately
        noise = np.random.normal(loc=0.5, scale=0.1, size=(num_freqs, total_samples)).astype(complex)
        if complex_sine:
            noise_imag = np.random.normal(loc=0.5, scale=0.1, size=(num_freqs, total_samples))
            # The += operator works perfectly here since noise is already complex
            noise += 1j * noise_imag
    else: 
        # Unconditionally set the zero array to complex
        noise = np.zeros((num_freqs, total_samples), dtype=complex)

    data = noise + cw_signal # Even real data will be cast to complex when added to complex noise
    
    return np.squeeze(data) # Remove any unnecessary dimensions for compatibility with downstream processing

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