from utilities import find_consts, gaussian_temp
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
params['points'] = 10000
params['steps'] = 100
params['delta_x'] = 0.001
#just set it below the CFL number for stability
# params['delta_t'] = params['delta_x'] ** 2 / 4.0
params['T_max'] = 0.1
params['t'] = np.linspace(0,
                          params['T_max'],
                          params['steps'])

params['include_exp'] = True
params['plot_every'] = 50

params['X'] = np.linspace(0,
                          params['points'] * params['delta_x'],
                          params['points'])
params['material'] = wetdiabase
params['stress'] = 100e6  # 100 MPa
params['length_scale'] = 1  # 1 meter, don't change this
params['temp_scale'], \
    params['time_scale'], \
    params['the_constant'] = find_consts(params['stress'],
                                         params['length_scale'],
                                         params['material'])
params['min_temp'] = 800 / params['temp_scale']  # Kelvins
params['temp_mass'] = 400 * 0.001 / params['temp_scale']
params['gaussian_width'] = 10000

# params['initial_temp'] = np.zeros((params['points'])) + params['min_temp']
#
# params['initial_temp'][params['points'] / 2 + 1] += \
#     params['temp_mass'] / params['delta_x']

params['initial_temp'] = gaussian_temp(params)
