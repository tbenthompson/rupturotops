from core.data import Data
from core.constants import consts
from material import wetdiabase
import numpy as np
from experiments.shear_heating import ShearHeating

params = Data()
params.material = wetdiabase
params.delta_t = 5.0 * consts.secs_in_a_year
params.t_max = 1000.0 * consts.secs_in_a_year
params.x_min = -500
params.x_max = 500
params.y_min = 0
params.y_max = 75000
params.x_points = 501
params.y_points = 100
params.background_temp = 773.0
params.temp_pulse_size = 0.0  # kelvins
params.initial_stress = 100e6
params.proj_name = 'test'
params.run_name = 'shear_heating'
params.plate_rate = (40.0 / 1.0e3) / consts.secs_in_a_year  # 40 mm/yr

params.source_friction_coeff = 0.005
params.source_term = params.source_friction_coeff * \
    params.material.shear_modulus * \
    params.plate_rate
print params.source_term

experiment = ShearHeating
