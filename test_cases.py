#!/bin/bash

import numpy as np
from ipdb import set_trace as stop
import test_signals as ts
import PFB

def test_dirac_combs(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=True):
    data = ts.generate_dirac_comb_signal(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, delta_period=delta_period, delta_start=delta_start, include_noise=include_noise)
    stop()

if __name__ == "__main__":
    n_taps = 4
    n_chan = 16
    n_windows = 100
    delta_period = 16
    delta_start = 1
    test_dirac_combs(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=False)