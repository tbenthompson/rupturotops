from material import wetdiabase

params = dict()
params['proj_name'] = 'heatcreep'
params['run_name'] = 'explorations'
params['material'] = wetdiabase
params['points'] = 5000
params['steps'] = 50.0
params['delta_x'] = 0.001
params['t_max'] = 5.0
params['plot_every'] = 100
params['include_exp'] = True

params['stress'] = 100e6  # 100 MPa
params['length_scale'] = 1  # 1 meter, don't change this
params['min_temp'] = 600  # Kelvins
params['temp_mass'] = 500
params['gaussian_width'] = 300

# params['initial_temp'] = np.zeros((params['points'])) + params['min_temp']
#
# params['initial_temp'][params['points'] / 2 + 1] += \
#     params['temp_mass'] / params['delta_x']
