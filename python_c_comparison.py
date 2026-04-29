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
    windows = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200] 
    M, P, freq = 4, 256, 1.0
    signal_type = "complex_phasors" 
    delta_period, delta_start = 257, 0
    nbit = 64
    include_noise = False

    cpp_executable = "/Users/hilays79/Fourier_Space/codes/PFB_cpp/objs/PFB_app" 

    # Expanded the table with exact column widths
    print(f"{'W':<7} | {'Py Time (s)':<12} | {'C++ Tot (s)':<12} | {'C++ Set (s)':<12} | {'C++ Exe (s)':<12} | {'C++ Set/Exec':<12} | {'Speedup':<9} | {'Max Diff':<10}")
    print("-" * 105)

    for W in windows:
        input_filepath = PFB.get_expected_input_filepath(
            signal_type, M, P, W, include_noise, freq, delta_period, delta_start
        )
        
        if not os.path.exists(input_filepath):
            gbd.create_binary_test_signals(
                n_taps=M, n_chan=P, n_windows=W, freq=freq,
                delta_period=delta_period, delta_start=delta_start,
                nbit=nbit, include_noise=include_noise, signal_type=signal_type
            )
        
        header, input_data = gbd.read_dada_file(input_filepath)
        
        py_start = time.perf_counter()
        py_out = PFB.pfb_spectrometer(input_data, n_taps=M, n_chan=P)
        py_time = time.perf_counter() - py_start

        result = subprocess.run([cpp_executable, str(W)], capture_output=True, text=True)
        
        cpp_total = None
        cpp_setup = None
        cpp_exec = None
        
        # Scrape all three times from the C++ output
        for line in result.stdout.split('\n'):
            if line.startswith("CPP_MATH_TIME:"):
                cpp_total = float(line.split(":")[1])
            elif line.startswith("CPP_SETUP_TIME:"):
                cpp_setup = float(line.split(":")[1])
            elif line.startswith("CPP_EXEC_TIME:"):
                cpp_exec = float(line.split(":")[1])
                
        if cpp_total is None:
            print(f"Error running C++ for W={W}.\n{result.stderr}\n{result.stdout}")
            continue

        cpp_output_filepath = input_filepath.replace("input_files", "output_files/c++")
        
        if os.path.exists(cpp_output_filepath):
            _, cpp_out = gbd.read_dada_file(cpp_output_filepath)
            max_diff = np.max(np.abs(np.squeeze(py_out) - np.squeeze(cpp_out)))
            diff_str = f"{max_diff:.2e}" 
        else:
            diff_str = "File Missing"
            
        speedup = py_time / cpp_total
        set_exec_ratio = cpp_setup / cpp_exec

        # Pre-format the speedup string so the f-string can pad it perfectly
        speedup_str = f"{speedup:.2f}x"
        
        # Print the fully broken-down results
        print(f"{W:<7} | {py_time:<12.6f} | {cpp_total:<12.6f} | {cpp_setup:<12.6f} | {cpp_exec:<12.6f} | {set_exec_ratio:<12.6f} | {speedup_str:<9} | {diff_str:<10}")
        
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