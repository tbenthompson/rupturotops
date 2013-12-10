from core.data import Data
from projection.controller import ProjController
from parameters.material import wetdiabase
from core.constants import consts
from core.debug import _DEBUG

params = Data()

#what material to use
params.material = wetdiabase

#time stepping
params.delta_t = 0.1 * consts.secs_in_a_year
params.t_max = 0.1 * consts.secs_in_a_year

#grid descriptors
params.x_min = -2.0e4
params.x_max = 2.0e4
params.y_min = 0.0
params.y_max = 2.0e5
params.x_points = 100
params.y_points = 1000
params.delta_x = (params.x_max - params.x_min) / params.x_points
params.delta_y = (params.y_max - params.y_min) / params.y_points


#initial temp description
params.background_temp = 960.0
params.temp_pulse_size = 0.0  # kelvins

#temperature source function
params.plate_rate = 1.6225265844296512e-10#(40.0 / 1.0e3) / consts.secs_in_a_year  # 40 mm/yr
params.source_friction_coeff = 0.005
params.source_term = params.source_friction_coeff * \
    params.material.shear_modulus * \
    params.plate_rate

#initial stress setup
params.initial_stress = 100.0e6
params.fault_slip = 2.0
params.fault_depth = 1.0e4
params.elastic_depth = 1.0e4
params.viscosity = 5.0e19

#administrative stuff -- where to store files?
params.proj_name = 'test'
params.run_name = 'visco_projection'

experiment = ProjController
