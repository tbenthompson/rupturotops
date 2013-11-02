from core.data_controller import DataController
from experiments.fenics.fenics_waves import Waves
from parameters.material import wetdiabase
from core.constants import consts
from math import sqrt

params = DataController()
params.material = wetdiabase
params.t_max = 100.0
params.x_min = -5.0
params.x_max = 5.0
params.y_min = -5.0
params.y_max = 5.0
params.x_points = 101
params.y_points = 101
delta_x = (params.x_max - params.x_min) / params.x_points
c = sqrt(params.material.shear_modulus / params.material.density)
params.delta_t = 0.1 * delta_x / c
print "Delta_t: " + str(params.delta_t)
params.proj_name = 'fenics'
params.run_name = 'wave_not_dg'

experiment = Waves
