#!/bin/bash

import numpy as np
from ipdb import set_trace as stop
import test_signals as ts
import PFB
import matplotlib.pyplot as plt
import scipy

def test_spectral_leakage_dirac_comb(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=True):
    """ Test spectral leakage in a PFB spectrometer by generating a dirac comb signal and analyzing the resulting spectrum. """
    data = ts.generate_dirac_comb_signal(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, delta_period=delta_period, delta_start=delta_start, include_noise=include_noise)

    stop()

def test_spectral_leakage_sine(n_taps, n_chan, n_windows, freq, include_noise=True, plot=True, complex_sine=False):
    """ Test spectral leakage in a PFB spectrometer by generating a sine wave signal and analyzing the resulting spectrum. """
    data = ts.generate_sine_signal(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, freq=freq, include_noise=include_noise, complex_sine=complex_sine)
    # Apply PFB spectrometer to the generated sine wave signal
    X_psd_PFB = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming")
    X_psd_brute = PFB.brute_force_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming")
    X_psd_fft = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="rectangular")
    x_psd_fft_windowed = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="hamming")
    if plot:
        plt.plot(PFB.db(X_psd_PFB)[0], label='PFB', alpha=1)
        plt.plot(PFB.db(X_psd_brute)[0]-2, label='Brute Force (shifted down by 2 dB for visibility)', alpha=0.5)
        plt.plot(PFB.db(X_psd_fft)[0]+2, label='FFT (shifted up by 2 dB for visibility)', alpha=1)
        plt.plot(PFB.db(x_psd_fft_windowed)[0]+4, label='FFT with Hamming Window (shifted up by 4 dB for visibility)', alpha=1)
        
        # Add some padding to the title to push it up above the new legend space
        plt.title('Spectral Leakage test \n Omega = {}'.format(freq), pad=75) 
        
        plt.ylim(-40, 40)
        plt.xlim(0, n_chan)
        plt.xlabel("Channel")
        plt.ylabel("Power [dB]")
        
        # Position the legend above the plot, centered
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
        
        # Ensure the expanded top area isn't cut off when rendering
        plt.tight_layout() 
        plt.show()

    stop()

def test_channel_response_sweep(n_taps, n_chan, n_windows, freqs, channels_to_plot=(0, 1, 2), include_noise=False, complex_sine=False):
    """
    Sweeps a sine wave across a range of frequencies and records the power
    response in specific PFB channels to visualise the filter shapes.
    """
    # Dictionary to hold the power values for each requested channel
    channel_powers = {ch: [] for ch in channels_to_plot}
    
    # Loop over each frequency in the array
    for freq in freqs:
        # 1. Generate the test signal using your framework generator
        data = ts.generate_sine_signal(n_taps=n_taps, 
                                    n_chan=n_chan, 
                                    n_windows=n_windows, 
                                    freq=freq, 
                                    include_noise=include_noise, 
                                    complex_sine=complex_sine)
        
        # 2. Pass the generated data through the PFB spectrometer.
        # Setting n_int = n_windows ensures we get exactly one integrated block back.
        X_psd = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming")
        
        # 3. Extract and store the power for the specific bins we care about
        for ch in channels_to_plot:
            channel_powers[ch].append(X_psd[0, ch])
            
    # 4. Plot the results
    plt.figure(figsize=(10, 6))
    
    for ch in channels_to_plot:
        plt.plot(freqs, PFB.db(channel_powers[ch]), label=f"Channel {ch}")
        
    # Vertical lines at the theoretical center frequencies of the channels
    # bin_center_freq = 2 * pi * k / P
    for ch in channels_to_plot:
        center_freq = 2 * np.pi * ch / n_chan
        if center_freq <= freqs[-1]: # Only plot the line if it falls within our sweep range
            plt.axvline(x=center_freq, color='gray', linestyle='--', alpha=0.5)

    plt.xlim(freqs[0], freqs[-1])
    # Note: What your snippet called 'period' is actually the angular frequency omega (rad/sample)
    plt.xlabel("Frequency $\omega$ [rad/sample]") 
    plt.ylabel("Power [dB]")
    plt.title("PFB Channel Frequency Response Sweep")
    # Position the legend completely outside the plot to the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_analytical_channel_response(n_taps, n_chan, channel=1, window_fn="hamming", plot=True):
    """
    Calculates the theoretical, unpadded frequency response of a specific PFB channel 
    directly from the filter tap coefficients.
    """
    M = n_taps
    P = n_chan
    num_taps = M * P

    # Generate the exact prototype filter tap coefficients
    win_coeffs = scipy.signal.get_window(window_fn, num_taps)
    sinc = scipy.signal.firwin(num_taps, cutoff=1.0/P, window="rectangular") 
    prototype_filter = win_coeffs * sinc
    
    # Normalize for processing gain to match empirical spectrometer power levels
    pg = np.sum(np.abs(prototype_filter)**2)
    prototype_filter /= pg**0.5

    n = np.arange(num_taps)

    # Take the standard, unpadded FFT (Length is exactly M * P)
    H_unpadded = np.fft.fft(prototype_filter)

    # Calculate zero-padded FFT for high-resolution smooth curve
    pad_factor = 16 
    N_pad = num_taps * pad_factor
    H_padded = np.fft.fft(prototype_filter, n=N_pad)

    H_db_unpadded = PFB.db(np.real(H_unpadded * np.conj(H_unpadded)))
    H_db_padded = PFB.db(np.real(H_padded * np.conj(H_padded)))

    # 5. Create frequency arrays and shift both arrays so 0 is in the center
    # Unpadded shift
    omega_unpadded = np.fft.fftshift(np.fft.fftfreq(num_taps)) * 2 * np.pi
    H_db_unpadded_shifted = np.fft.fftshift(H_db_unpadded)
    
    # Padded shift
    omega_padded = np.fft.fftshift(np.fft.fftfreq(N_pad)) * 2 * np.pi
    H_db_padded_shifted = np.fft.fftshift(H_db_padded)

    if plot:
        plt.figure(figsize=(10, 6))

        # Plot the smooth zero-padded curve first (so it sits behind the discrete points)
        plt.plot(omega_padded, H_db_padded_shifted, linestyle='-', color='blue', alpha=0.6, label="Theoretical Ch 0 (Zero-Padded)")

        # Plot the discrete unpadded points
        # Changed linestyle to 'none' so the polygonal lines don't clutter the smooth curve, 
        # but you can change it back to '-' if you prefer seeing the connections.
        plt.plot(omega_unpadded, H_db_unpadded_shifted, marker='.', linestyle='none', color='orange', label="Theoretical Ch 0 (Unpadded)")
        
        # Draw a vertical line at the theoretical center frequency for reference
        center_freq = 0
        plt.axvline(x=center_freq, color='gray', linestyle='--', alpha=0.5, label="Center Freq Ch 0")

        # Zoom in to a width of roughly two channels on either side (+/- 2 bins)
        channel_width = 2 * np.pi / P
        plt.xlim(-2 * channel_width, 2 * channel_width) 
        plt.ylim(-40, 40)
        
        plt.xlabel(r"Frequency $\omega$ [rad/sample]")
        plt.ylabel("Power [dB]")
        plt.title("Analytical PFB Channel 0 Response")
        
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    return omega_padded, H_db_padded_shifted

if __name__ == "__main__":
    n_taps = 4
    n_chan = 256
    n_windows = 100
    delta_period = 16
    delta_start = 1
    freq=1
    # freq = (64)*np.pi/256
    # test_spectral_leakage_dirac_comb(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=False)
    # test_spectral_leakage_sine(n_taps, n_chan, n_windows, freq, include_noise=False, complex_sine=False)
    # Angular frequency sweep from 0 to slightly past bin 2's center
    highest_channel = 16
    omegas = np.linspace(0.002, highest_channel*1.1*2*np.pi/n_chan, 1000)
    test_channel_response_sweep(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, freqs=omegas, channels_to_plot=np.arange(0, highest_channel),include_noise=False, complex_sine=False)
    # get_analytical_channel_response(n_taps=n_taps, n_chan=n_chan, channel=0, window_fn="hamming", plot=True)