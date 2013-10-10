import numpy as np
from math import exp, sqrt, pi
# from pdb import set_trace as _DEBUG
import scipy.integrate
import os, shutil


def gaussian_temp(params):
    temp = (params['temp_mass'] / sqrt(pi)) * \
        np.exp(-params['gaussian_width'] * params['X'] ** 2)
    temp += params['min_temp']
    return temp.T


def test_gaussian_temp():
    params = dict()
    params['X'] = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    params['temp_mass'] = 1
    params['min_temp'] = 0
    params['gaussian_width'] = 1
    temp = gaussian_temp(params)
    assert temp[1] < temp[2]
    assert temp[2] > temp[3]
    assert temp[3] > temp[4]
    #distance = 0.5 , distance ** 2 = 0.25
    correct = (0.7788007830714049 / 1.7724538509055159)
    assert (temp[0] - correct) < 0.000001

    params['temp_mass'] = sqrt(pi)
    temp2 = gaussian_temp(params)
    assert temp2[2] == 1

    params['gaussian_width'] = 2
    temp3 = gaussian_temp(params)
    assert temp3[0] == exp(-0.5)

    params['min_temp'] = 1
    params['temp_mass'] = 2 * sqrt(pi)
    temp4 = gaussian_temp(params)
    assert temp4[2] == 3


def calc_strain(temp, data):
    tointegrate = data['material']['creepconstant'] * data['time_scale'] * \
        data['stress'] ** data['material']['stressexponent'] * \
        np.exp(-1 / temp)
    strain = scipy.integrate.cumtrapz(tointegrate,
                                      dx=data['t_max'] / data['steps'])
    return strain


def test_calc_stran():
    temp = np.array([-1.0, -1.0, -1.0])
    data = dict()
    data['stress'] = 1.0
    data['material'] = dict(creepconstant=1.0, stressexponent=1.0)
    data['time_scale'] = 1.0
    data['steps'] = 2.0
    data['t_max'] = 2.0
    strain = calc_strain(temp.T, data)
    assert (strain == [exp(1), exp(2)]).all


def find_consts(stress, length_scale, rock_params):
    temp_scale = rock_params['activationenergy'] / rock_params['R']
    time_scale = length_scale ** 2 / rock_params['thermaldiffusivity']
    numer = (time_scale * rock_params['creepconstant']
             * stress ** (rock_params['stressexponent'] + 1))
    denom = (temp_scale * rock_params['specificheat'] * rock_params['density'])
    return temp_scale, time_scale, numer / denom


def lowest_open_space(prefix, runs_list):
    similar_runs = [run.replace(prefix, '') for run in
                    runs_list if run.startswith(prefix)]
    #chopped runs should only be numbers now, throw out the others
    final_runs = [int(run) for run in similar_runs if run.isdigit()]
    if len(final_runs) == 0:
        return prefix + '0'
    max_run = max(final_runs)
    return prefix + str(max_run + 1)


def test_lowest_open_space():
    result = lowest_open_space('abc', ['abc1', 'abc2', 'abc3'])
    assert result == 'abc4'

    result = lowest_open_space('abc', [])
    assert result == 'abc0'

    result = lowest_open_space('abc', ['def'])
    assert result == 'abc0'


def assign_data_location(params):
    root = '../../data/'
    folder_name = root + params['proj_name']
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    existing_runs = os.listdir(folder_name)
    new_run_folder_name = folder_name + '/' + \
        lowest_open_space(params['run_name'], existing_runs)
    os.mkdir(new_run_folder_name)
    params['data_loc'] = new_run_folder_name


def test_assign_data_location():
    params = dict()
    params['proj_name'] = 'test'
    params['run_name'] = 'test'
    assign_data_location(params)
    dir_exists = os.path.exists('../../data/test/test0')
    assert dir_exists
    shutil.rmtree('../../data/test/test0')


def finish_calc_params(params):
    assign_data_location(params)

    params['t'] = np.linspace(0,
                              params['t_max'],
                              params['steps'])

    half_width = params['points'] * params['delta_x'] / 2.0
    params['X'] = np.linspace(-half_width,
                              half_width,
                              params['points'])
    params['temp_scale'], \
        params['time_scale'], \
        params['the_constant'] = find_consts(params['stress'],
                                             params['length_scale'],
                                             params['material'])
    params['min_temp'] = params['min_temp'] / params['temp_scale']  # Kelvins
    params['temp_mass'] = params['temp_mass'] / params['temp_scale']
    params['initial_temp'] = gaussian_temp(params)
