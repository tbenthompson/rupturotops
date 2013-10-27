from core.data_controller import DataController
from experiments.dg_waves import DGWaves
from parameters.material import wetdiabase
from core.constants import consts
from math import sqrt

params = DataController()
params.material = wetdiabase
params.t_max = 100.0
params.x_min = -50000.0
params.x_max = 50000.0
params.y_min = 0.0
params.y_max = 15000.0
params.x_points = 151
params.y_points = 61
delta_x = (params.x_max - params.x_min) / params.x_points
c = sqrt(params.material.shear_modulus / params.material.density)
params.delta_t = 4 * delta_x / c
print "Delta_t: " + str(params.delta_t)
params.proj_name = 'fenics'
params.run_name = 'wave'
experiment = DGWaves(params)
