from math import pi,factorial   
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as pyp

def constant_slip_constant_shear_modulus_viscoelastic(x, t, alpha, a):
    def E(n):
        retval = 0
        for m in range(1, n+1):
            retval += ((-a * t)**(n-m))/factorial(n-m)
        return retval
    result = 0.0
    addthis = 0.0
    n = 1
    while True:
        addthis = (1 - (np.exp(a * t) * E(n))) * (dimensionless_arctan(2 * n) + dimensionless_arctan(-2 * n))
        if addthis == 0 or abs(addthis/result) < 0.01:
            break
        result += addthis
        n += 1
    result += dimensionless_arctan(0)
    result /= pi
    return result
