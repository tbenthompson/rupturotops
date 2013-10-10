from utilities import finish_calc_params
import numpy as np

#ALL IN STANDARD SI UNITS

#wet diabase from takeuchi, fialko
wetdiabase = dict()
wetdiabase['density'] = 2850  # kg/m^3
wetdiabase['specificheat'] = 1000  # J/kgK
wetdiabase['R'] = 8.314  # J/molK
wetdiabase['activationenergy'] = 2.6e5  # J/mol
wetdiabase['stressexponent'] = 3.4
wetdiabase['creepconstant'] = 2.2e-4 * 10 ** (-6 * 3.4)  # (Pa^-n)/sec
wetdiabase['thermaldiffusivity'] = 7.4e-7  # m^2/sec
wetdiabase['youngsmodulus'] = 80e9  # Pa

params = dict()
params['proj_name'] = 'heatcreep'
params['run_name'] = 'explorations'
params['material'] = wetdiabase
params['points'] = 500
params['steps'] = 50.0
params['delta_x'] = 0.0001
params['t_max'] = 5.0e-7
params['plot_every'] = 100
params['include_exp'] = True

params['stress'] = 500e7    # 100 MPa
params['length_scale'] = 1  # 1 meter, don't change this
params['min_temp'] = 900    # Kelvins
params['temp_mass'] = 500 * 1.5
params['gaussian_width'] = 300

# params['initial_temp'] = np.zeros((params['points'])) + params['min_temp']
#
# params['initial_temp'][params['points'] / 2 + 1] += \
#     params['temp_mass'] / params['delta_x']
