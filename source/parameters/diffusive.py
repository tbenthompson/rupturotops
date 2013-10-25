from core.data_controller import DataController
import numpy as np
from material import wetdiabase
from experiments.diffusive import Diffusive
from core.constants import consts

params = DataController()
params.material = wetdiabase
params.proj_name = 'diffusive'
params.run_name = 'test'
params.low_x = 0.0
params.high_x = 0.1
params.low_t = 0.001 * consts.secs_in_a_year
params.high_t = 1000000.0 * consts.secs_in_a_year
params.x_count = 25
params.t_count = 100
params.x_domain = np.linspace(params.low_x, params.high_x, params.x_count)
params.t_domain = np.linspace(params.low_t, params.high_t, params.t_count)
params.time_scale = 100.0 * consts.secs_in_a_year  # 100 years
params.init_temp = 350.0 + 273.0  # initial temperature
params.delta_t = 2.5  # total increase in temperature for an event
params.stress = 100.0e6  # MPa

experiment = Diffusive(params)
