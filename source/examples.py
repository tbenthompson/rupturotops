import numpy as np
from math import pi
import matplotlib.pyplot as pyp
from pdb import set_trace as _DEBUG

import utilities
import full_viscoelastic

def standard_params(slip):
    x = np.arange(0.05, 10.0, 0.05)
    t = np.arange(0.0, 5.0)
    alpha = 0.0
    u_t = map(lambda t_in: full_viscoelastic.solution(slip, x, t_in, alpha), t)
    utilities.plot_time_series_1D(x, u_t, t)

if __name__ == "__main__":
    standard_params(lambda z: 1 - np.sin(pi * z))
    # standard_params(lambda z: np.sin(pi * z))
    # standard_params(lambda z: 1 if z > 0.5 else 0)
