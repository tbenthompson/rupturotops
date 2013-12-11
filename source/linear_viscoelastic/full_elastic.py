from math import pi
from utilities import A_integral
import scipy.integrate as integrate
import numpy as np
from core.debug import _DEBUG
assert(np)

#derived by differentiating the slip for slips delta z apart
def surface_elastic_half_space(slip_distribution, x):
    A = A_integral(slip_distribution, x, 1, 0)
    # print A
    return (x/pi) * A

#based on the infinite images solution
def surface_two_layer_elastic(slip_distribution, alpha,
                              beta, x, truncation = 10):
    outer_factor = x/pi
    main_fault_term = A_integral(slip_distribution, x, alpha, 0)
    images = 0
    for i in range(1, truncation + 1):
        comp1 = A_integral(slip_distribution, x, alpha, 2 * i)
        comp2 = A_integral(slip_distribution, x, alpha, -2 * i)
        images += (beta ** i) * (comp1 + comp2)
    result = outer_factor * (main_fault_term + images)
    return result


def A_integral_depth(slip_distribution, x, y, D, F):
    @np.vectorize
    def integration_controller(x_in, y_in):
        def integrand(z, x_inside):
            denom = x_inside ** 2 + (y_in + z * F) ** 2
            return slip_distribution(z) / denom
        return integrate.quad(integrand, 0, D, args = (x_in,))[0]
    return integration_controller(x, y)


def elastic_half_space(slip_distribution, D, x, y):
    # This should be shifted to the non-dimensional form of the
    # above equations
    factor = x / (2 * pi)
    main = A_integral_depth(slip_distribution, x, y, D, 1)
    image = A_integral_depth(slip_distribution, x, y, D, -1)
    return factor * (main + image)


# For stress
def Szx_integral_depth(slip_distribution, x, y, D, F):
    @np.vectorize
    def integration_controller(x_in, y_in):
        def integrand(z, x_inside):
            numer = ((y_in + F * z) ** 2 - x_inside**2)
            denom = ((y_in + F * z) ** 2 + x_inside**2) ** 2
            return numer * slip_distribution(z) / denom
        return integrate.quad(integrand, 0, D, args = (x_in,))[0]
    return integration_controller(x, y)

def Szy_integral_depth(slip_distribution, x, y, D, F):
    @np.vectorize
    def integration_controller(x_in, y_in):
        def integrand(z, x_inside):
            numer = 2 * x_inside * (y_in + F * z)
            denom = ((y_in + F * z) ** 2 + x_inside**2) ** 2
            return numer * slip_distribution(z) / denom
        return integrate.quad(integrand, 0, D, args = (x_in,))[0]
    return integration_controller(x, y)

def elastic_half_space_stress(slip_distribution, D, x, y, mu):
    factor = mu / (2 * pi)
    Szx_main = Szx_integral_depth(slip_distribution, x, y, D, 1)
    Szx_image = Szx_integral_depth(slip_distribution, x, y, D, -1)
    Szx = factor * (Szx_main + Szx_image)

    Szy_main = Szy_integral_depth(slip_distribution, x, y, D, 1)
    Szy_image = Szy_integral_depth(slip_distribution, x, y, D, -1)
    Szy = -factor * (Szy_main + Szy_image)
    return Szx, Szy

def test_elastic_half_space():
    s = lambda z: 1.0
    y = 0.0
    x = np.linspace(-5, 5, 10)
    D = 1.0
    u_new = elastic_half_space(s, D, x, y)
    u_test = surface_elastic_half_space(s, x)
    np.testing.assert_almost_equal(u_new, u_test)


