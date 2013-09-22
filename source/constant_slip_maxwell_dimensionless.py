from math import pi,factorial   
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as pyp

from utilities import dimensionless_arctan

#this assumes constant slip and constant shear modulus between two layers, an elastic top layer and a viscoelastic bottom layer
def solution(x, t, alpha):
    def E(n):
        retval = 0l
        for m in range(1, n+1):
            retval += ((t)**(n-m))/factorial(n-m)
        return retval
    result = 0.0
    addthis = 0.0
    n = 1
    while True:
        addthis = (1 - (np.exp(-t) * E(n))) * (dimensionless_arctan(x, alpha, 2 * n) + dimensionless_arctan(x, alpha, -2 * n))
        if (n != 1) and ((addthis == 0).all() or (abs(addthis/result) < 0.01).all()):
            break
        result += addthis
        n += 1
    result += dimensionless_arctan(x, alpha, 0)
    result /= pi
    return result
