from core.data import Data
from experiments.controller import Controller
from parameters.material import wetdiabase
from core.constants import consts
from experiments import wave_forms
assert(consts)
from core.debug import _DEBUG
assert(_DEBUG)
import numpy as np

# Define the standard parameter data structure.
params = Data()

# What material should we use? Look in the parameters/material file
# to see the options and to see how to define a new material.
params.material = wetdiabase


# Setup the solution domain, first we define the cell spacings
delta_x = []
for i in range(1000):
    if i % 10 == 0:
        delta_x.append(0.002)
    else:
        delta_x.append(0.004)
params.delta_x = np.array(delta_x)

# plotting parameters
params.plotter = Data()
params.plotter.always_plot = False
params.plotter.never_plot = True
params.plotter.plot_interval = 0.5


params.t_max = 2.0
params.analytical = wave_forms.sin_4

# Define project and run parameters in order to save the results to
# the proper data folder.
params.proj_name = 'fvm'
params.run_name = 'development'

# Define the set of experiments to be run
experiment = Controller
