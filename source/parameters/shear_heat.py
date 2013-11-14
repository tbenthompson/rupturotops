from core.data import Data
from core.constants import consts
from material import wetdiabase
import numpy as np
from experiments.shear_heating import ShearHeating

params = Data()
params.material = wetdiabase
params.x_points = 101  # must be odd
params.x = np.linspace(-20000, 20000, params.x_points)  # meters
params.t_max = 100
params.t = np.linspace(0, consts.secs_in_a_year * params.t_max, 100)  # time
params.temp_pulse_size = 3.0  # kelvins
# params.initial_temp_start_time = 1.0e6
# params.initial_temp = 500 + params.temp_pulse_size / np.sqrt(
#     4 * np.pi * params.material.thermal_diffusivity * params.initial_temp_start_time) * \
#     np.exp(-(params.x ** 2 / (4 * params.material.thermal_diffusivity * params.initial_temp_start_time)))
params.initial_temp = 600.0 * np.ones(params.x.shape)
params.plate_rate = (40.0 / 1.0e3) / consts.secs_in_a_year  # 40 mm/yr
params.source_term = np.zeros(params.x.shape)
params.source_friction_coeff = 0.005
params.source_term[(params.x_points - 1) / 2] = params.source_friction_coeff * \
    params.material.shear_modulus * \
    params.plate_rate / \
    (params.material.specific_heat * params.material.density * 1)
params.low_temp = 500.0  # kelvins
params.delta_temp = 1000.0
params.stress = 100.0e6  # pascals
params.proj_name = 'test'
params.run_name = 'shear_heating'
params.plot_init_cond = False
params.plot_temp_time = True
params.plot_strain_time = False
params.plot_every = 10  # time steps

experiment = ShearHeating
