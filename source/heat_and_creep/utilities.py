import numpy as np
from math import exp
from pdb import set_trace as _DEBUG


def gaussian_temp(points, min, max, width):
    x = (np.linspace(0, 1, points) - 0.5)
    _DEBUG()
    temp = max * np.exp(-x ** 2 / width) 
    temp += min
    return temp


def test_gaussian_temp():
    temp = gaussian_temp(5, 0, 1, 1)
    assert temp[1] < temp[2]
    assert temp[2] > temp[3]
    assert temp[3] > temp[4]
    #distance = 0.5 , distance ** 2 = 0.25
    assert temp[0] == exp(-0.25)
