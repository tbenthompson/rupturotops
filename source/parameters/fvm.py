from core.data_controller import DataController
from experiments.fvm import FVM
from parameters.material import wetdiabase
from core.constants import consts
assert(consts)
from core.debug import _DEBUG
assert(_DEBUG)
import numpy as np


def positions_from_spacing(spacings):
    """
    Sum the spacings to get cell centers,
    but the cell center is at the halfway point, so we have to
    subtract half of delta_x again.
    """
    x = np.cumsum(spacings)
    x = x - spacings / 2.0
    return x
# Define the standard parameter data structure.
params = DataController()

# What material should we use? Look in the parameters/material file
# to see the options and to see how to define a new material.
params.material = wetdiabase

# The length of time for which to run the simulation.
params.t_max = 100.0

# Setup the solution domain, first we define the cell spacings
params.delta_x = np.ones(101) * 0.1

# and then we define the cell positions by summing the spacings.
params.x = positions_from_spacing(params.delta_x)

# Set the timestep based on the CFL condition
# Wave speed 1.0
c = 1.0
CFL_approx = 0.5
params.delta_t = CFL_approx * params.delta_x / c

# Define project and run parameters in order to save the results to
# the proper data folder.
params.proj_name = 'fvm'
params.run_name = 'development'

# Define the set of experiments to be run
experiment = FVM

#-----------------------------------------
# TESTS
#-----------------------------------------


def test_cell_spacing():
    spacings = np.array([0.5, 1.0, 0.1, 2.0])
    correct = np.array([0.25, 1.0, 1.55, 2.6])
    x = positions_from_spacing(spacings)
    assert(len(x) == 4)
    assert((x == correct).all())
