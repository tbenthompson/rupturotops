from core.data_controller import DataController
import numpy as np
from material import wetdiabase

params = DataController()
params.material = wetdiabase
params.low_x = 0.0
params.high_x = 1.0
params.low_t = 0.01
params.high_t = 1.0
params.x_count = 500
params.t_count = 500
params.x_domain = np.linspace(params.low_x, params.high_x, params.x_count)
params.t_domain = np.linspace(params.low_t, params.high_t, params.t_count)
params.time_scale = 100 * 365 * 24 * 3600  # 100 years
params.init_temp = 623  # initial temperature
params.delta_t = 3  # total increase in temperature for an event
