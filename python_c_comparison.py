#!/usr/bin/env python3

import os
import numpy as np
import generate_binary_data as gbd  # Importing your parser file
from ipdb import set_trace as stop

def get_output_filepath(language, signal_type, n_taps, n_chan, n_windows, include_noise, freq=None, delta_period=None, delta_start=None):
    """Helper to construct the expected output file path for either Python or C++."""
    # Select the base directory based on the language
    savepath_base = f"/Users/hilays79/Fourier_Space/Data/output_files/{language}/"
    
    if signal_type in ["sinusoidals", "complex_phasors"]:
        filenamestart = f"{signal_type}_freq{freq}_M{n_taps}_P{n_chan}_W{n_windows}_noise{include_noise}"
    elif signal_type == "dirac_deltas":
        filenamestart = f"{signal_type}_d{delta_period}_s{delta_start}_noise{include_noise}"
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
        
    return os.path.join(savepath_base, signal_type, f"{filenamestart}.dada")

def load_comparison_arrays(signal_type, n_taps, n_chan_out, n_windows, include_noise=False, freq=None, delta_period=None, delta_start=None):
    """
    Finds, parses, and returns both the Python and C++ output arrays for a specific test run.
    """
    
    # 1. Build the paths
    py_path = get_output_filepath(
        "python", signal_type, n_taps, n_chan_out, n_windows, include_noise, freq, delta_period, delta_start
    )
    cpp_path = get_output_filepath(
        "c++", signal_type, n_taps, n_chan_out, n_windows, include_noise, freq, delta_period, delta_start
    )
    
    # 2. Check if they exist before parsing
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python output missing. Did you run the Python pipeline?\nExpected: {py_path}")
    if not os.path.exists(cpp_path):
        raise FileNotFoundError(f"C++ output missing. Did you run the C++ pipeline?\nExpected: {cpp_path}")
        
    # 3. Read the data (ignoring the headers for the return statement, but you could return them if needed)
    print(f"Loading Python data from: .../python/{os.path.basename(py_path)}")
    py_header, py_data = gbd.read_dada_file(py_path)
    
    print(f"Loading C++ data from:    .../c++/{os.path.basename(cpp_path)}")
    cpp_header, cpp_data = gbd.read_dada_file(cpp_path)
    
    return py_data, cpp_data

def calculate_comparison_metrics(py_array, cpp_array):
    """Calculate and print comparison metrics between the two arrays."""
    if py_array.shape != cpp_array.shape:
        raise ValueError(f"Shape mismatch: Python array shape {py_array.shape} vs C++ array shape {cpp_array.shape}")
    
    # Calculate element-wise differences
    # if complex part of py_array is zero, make it explicitly real
    if np.iscomplexobj(py_array) and np.all(py_array.imag == 0):
        py_array = py_array.real
    difference = py_array - cpp_array
    abs_difference = np.abs(difference)
    # Metrics
    max_diff = np.max(abs_difference)
    mean_diff = np.mean(abs_difference)
    mse = np.mean(difference**2)
    
    print(f"Max Absolute Difference: {max_diff}")
    print(f"Mean Absolute Difference: {mean_diff}")
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    # Example usage based on your recent test parameters
    M, P, W, freq = 4, 256, 100, 1
    signal_type = "complex_phasors"
    
    py_array, cpp_array = load_comparison_arrays(
        signal_type=signal_type,
        n_taps=M,
        n_chan_out=P,
        n_windows=W,
        include_noise=False,
        freq=freq
    )
    calculate_comparison_metrics(py_array, cpp_array)
    stop()