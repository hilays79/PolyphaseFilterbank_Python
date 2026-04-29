#!/usr/bin/env python3

import os
import numpy as np
import generate_binary_data as gbd
from ipdb import set_trace as stop
import matplotlib.pyplot as plt
import time
import subprocess
import PFB

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

def run_benchmark():
    # 1. Setup Benchmark Parameters
    windows = [100, 200, 400, 800, 1600, 3200, 6400, 12800] # You can adjust this list based on your testing needs
    M, P, freq = 4, 256, 1.0
    signal_type = "complex_phasors" # Match what you set in C++
    delta_period, delta_start = 257, 0
    nbit = 64
    include_noise = False

    # ---------------------------------------------------------
    # ABSOLUTE PATH TO C++ EXECUTABLE
    # ---------------------------------------------------------
    cpp_executable = "/Users/hilays79/Fourier_Space/codes/PFB_cpp/objs/PFB_app" 

    # Updated table header to include Max Diff
    print(f"{'Windows (W)':<12} | {'Python Time (s)':<17} | {'C++ Time (s)':<17} | {'Speedup':<9} | {'Max Diff':<12}")
    print("-" * 80)

    for W in windows:
        # --- A. FILE PREPARATION ---
        input_filepath = PFB.get_expected_input_filepath(
            signal_type, M, P, W, include_noise, freq, delta_period, delta_start
        )
        
        # Auto-generate the binary file if it doesn't exist for this W
        if not os.path.exists(input_filepath):
            gbd.create_binary_test_signals(
                n_taps=M, n_chan=P, n_windows=W, freq=freq,
                delta_period=delta_period, delta_start=delta_start,
                nbit=nbit, include_noise=include_noise, signal_type=signal_type
            )
        
        # --- B. PYTHON TIMING ---
        header, input_data = gbd.read_dada_file(input_filepath)
        
        py_start = time.perf_counter()
        # Strictly timing the math execution
        py_out = PFB.pfb_spectrometer(input_data, n_taps=M, n_chan=P)
        py_time = time.perf_counter() - py_start

        # --- C. C++ TIMING ---
        # Pass W as a command line argument
        result = subprocess.run([cpp_executable, str(W)], capture_output=True, text=True)
        
        cpp_time = None
        for line in result.stdout.split('\n'):
            if line.startswith("CPP_MATH_TIME:"):
                cpp_time = float(line.split(":")[1])
                break
                
        if cpp_time is None:
            print(f"Error running C++ for W={W}. Did you run 'make' recently?\n{result.stderr}\n{result.stdout}")
            continue

        # --- D. COMPARE ACCURACY ---
        # Predict where C++ saved the file using a quick string replacement
        cpp_output_filepath = input_filepath.replace("input_files", "output_files/c++")
        
        if os.path.exists(cpp_output_filepath):
            _, cpp_out = gbd.read_dada_file(cpp_output_filepath)
            
            # Squeeze removes any empty 1D axes (like polarization) so the arrays align cleanly
            max_diff = np.max(np.abs(np.squeeze(py_out) - np.squeeze(cpp_out)))
            
            # Format in scientific notation (e.g., 1.25e-14)
            diff_str = f"{max_diff:.2e}" 
        else:
            diff_str = "File Missing"
            
        # --- E. RESULTS ---
        speedup = py_time / cpp_time
        print(f"{W:<12} | {py_time:<17.6f} | {cpp_time:<17.6f} | {speedup:.2f}x{' ':<4} | {diff_str:<12}")
        
if __name__ == "__main__":
    # Example usage based on your recent test parameters
    M, P, W, freq = 4, 256, 100, 1
    delta_period, delta_start = 257, 0
    include_noise = False
    # signal_type = "sinusoidals" # Can be "sinusoidals", "complex_phasors", or "dirac_deltas"
    # signal_type = "dirac_deltas"
    signal_type = "complex_phasors"
    
    py_array, cpp_array = load_comparison_arrays(
        signal_type=signal_type,
        n_taps=M,
        n_chan_out=P,
        n_windows=W,
        include_noise=False,
        freq=freq,
        delta_period=delta_period,
        delta_start=delta_start
    )
    calculate_comparison_metrics(py_array, cpp_array)
    run_benchmark()
    stop()