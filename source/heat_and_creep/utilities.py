import numpy as np
from math import exp, sqrt, pi
# from pdb import set_trace as _DEBUG
import scipy.integrate


def gaussian_temp(params):
    minx = params['X'][0]
    maxx = params['X'][-1]
    midpt = (maxx - minx) / 2.0
    x = (np.linspace(minx, maxx, len(params['X'])) - midpt)
    temp = (params['temp_mass'] / (params['delta_x'] * sqrt(pi))) * \
        np.exp(-(params['delta_x'] * x ** 2) * params['gaussian_width'])
    temp += params['min_temp']
    return temp.T


def test_gaussian_temp():
    X = [0, 0.25, 0.5, 0.75, 1.0]
    temp = gaussian_temp(X, 0, sqrt(pi), 1)
    assert temp[1] < temp[2]
    assert temp[2] > temp[3]
    assert temp[3] > temp[4]
    #distance = 0.5 , distance ** 2 = 0.25
    assert temp[0] == exp(-0.25)

    temp2 = gaussian_temp(X, 1, sqrt(pi), 1)
    assert temp2[2] == 2

    temp3 = gaussian_temp(X, 0, sqrt(pi), 2)
    assert temp3[0] == exp(-0.5)

    temp4 = gaussian_temp(X, 1, 2 * sqrt(pi), 1)
    assert temp4[2] == 3


def calc_strain(temp, data):
    tointegrate = data['material']['creepconstant'] * data['time_scale'] * \
        data['stress'] ** data['material']['stressexponent'] * \
        np.exp(-1 / temp)
    strain = scipy.integrate.cumtrapz(tointegrate, dx=data['T_max'] / data['steps'])
    return strain


def test_calc_stran():
    temp = np.array([-1.0, -1.0, -1.0])
    data = dict()
    data['stress'] = 1.0
    data['material'] = dict(creepconstant=1.0, stressexponent=1.0)
    data['time_scale'] = 1.0
    data['steps'] = 2.0
    data['T_max'] = 2.0
    strain = calc_strain(temp.T, data)
    assert (strain == [exp(1), exp(2)]).all


def find_consts(stress, length_scale, rock_params):
    temp_scale = rock_params['activationenergy'] / rock_params['R']
    time_scale = length_scale ** 2 / rock_params['thermaldiffusivity']
    numer = (time_scale * rock_params['creepconstant']
             * stress ** (rock_params['stressexponent'] + 1))
    denom = (temp_scale * rock_params['specificheat'] * rock_params['density'])
    return temp_scale, time_scale, numer / denom
