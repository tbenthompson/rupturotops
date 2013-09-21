from math import pi,factorial   
import numpy as np
import scipy.integrate as integrate

from utilities import dimensionless_arctan

def image_coeff(n):
    retval = 0
    for m in range(1, n+1):
        retval += ((-a * t)**(n-m))/factorial(n-m)
    retval *= np.exp(a * t)
    retval += 1
    return retval


def constant_slip_constant_shear_modulus_viscoelastic(x, t, alpha, a):
    result = 0.0
    addthis = 0.0
    n = 1
    while True:
        addthis = (1 - (np.exp(a * t) * E(n))) * (dimensionless_arctan(x, alpha, 2 * n) + dimensionless_arctan(x, alpha, -2 * n))
        if (n != 1) and (addthis == 0 or abs(addthis/result) < 0.01):
            break
        result += addthis
        n += 1
    result += dimensionless_arctan(x, alpha, 0)
    result /= pi
    return result
