from core.data_controller import DataController
from experiments.fenics.fenics_shear_heat import ShearHeatingFenics
from parameters.material import wetdiabase
from core.constants import consts

params = DataController()
params.material = wetdiabase
params.delta_t = 0.1 * consts.secs_in_a_year
params.t_max = 1 * consts.secs_in_a_year
params.x_min = -100
params.x_max = 100
params.x_points = 101
params.temp_pulse_size = 3.0  # kelvins
params.stress = 100e6
params.low_temp = 900.0  # kelvins
params.delta_temp = 1000.0
params.initial_temp_start_time = 1.0e7
params.proj_name = 'fenics'
params.run_name = 'shear_heating'
params.plate_rate = (40.0 / 1.0e3) / consts.secs_in_a_year  # 40 mm/yr
params.source_friction_coeff = 0.001
params.source_term = params.source_friction_coeff * \
    params.material.shear_modulus * \
    params.plate_rate / \
    (params.material.specific_heat * params.material.density * 1)
print params.source_term

experiment = ShearHeatingFenics
