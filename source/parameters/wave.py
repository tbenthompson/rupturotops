from core.data_controller import DataController
from experiments.waves import Waves
from parameters.material import wetdiabase
from core.constants import consts

params = DataController()
params.material = wetdiabase
params.delta_t = 0.005
params.t_max = 100.0
params.x_min = -5000.0
params.x_max = 5000.0
params.x_points = 101

params.proj_name = 'fenics'
params.run_name = 'wave'
experiment = Waves(params)
