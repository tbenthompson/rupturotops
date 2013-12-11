from math import pi, factorial
import numpy as np

# displacements

# all the functions for the maxwell viscoelastic solution in segall


def _surface_G(x, D, H):
    return np.arctan(D / x)


def _surface_F(x, D, H, n):
    retval = np.arctan((2 * x * D) / (x ** 2 + (2 * n * H) ** 2 - D ** 2))
    # print retval
    return retval


def _E(time_scale, n):
    retval = 0
    for j in range(1, n + 1):
        # print str(j) + " " + str(n)
        # print time_scale
        retval += ((time_scale) ** (n - j)) / factorial(n - j)
    return retval


def surface_solution(x, t, t_r, D, H, s, length_of_sum=20):
    u_3 = 0
    for i in range(length_of_sum):
        u_3 += (1 - (np.exp(-t / t_r) * _E(t / t_r, i + 1))) * \
            _surface_F(x, D, H, i + 1)
    u_3 += _surface_G(x, D, H)
    u_3 *= s / pi
    return u_3


def surface_velocity_solution(x, t, t_r, D, H, s, length_of_sum=20):
    v_3 = 0
    for i in range(length_of_sum):
        term = (((t / t_r) ** i) / factorial(i)) * _surface_F(x, D, H, i + 1)
        v_3 += term
    v_3 *= np.exp(-t / t_r)
    v_3 *= (s / (pi * t_r))
    return v_3

def _upper_F(x, y, D, H, n):
    term1 = np.arctan(x / (y - 2 * n * H - D))
    term2 = np.arctan(x / (y + 2 * n * H - D))
    term3 = -np.arctan(x / (y - 2 * n * H + D))
    term4 = -np.arctan(x / (y + 2 * n * H + D))
    return term1 + term2 + term3 + term4

def _lower_F(x, y, D, H, n):
    term1 = np.arctan(x / (y - 2 * n * H - D))
    term2 = -np.arctan(x / (y - 2 * n * H + D))
    return term1 + term2

def _depth_F(x, y, D, H, n):
    retval = np.where(y > H, _upper_F(x, y, D, H, n), _upper_F(x, y, D, H, n))
    return retval

def velocity_solution(x, y, t, t_r, D, H, s, length_of_sum=20):
    v_3 = 0
    for i in range(length_of_sum):
        term = (((t / t_r) ** i) / factorial(i)) * _depth_F(x, y, D, H, i + 1)
        v_3 += term
    v_3 *= np.exp(-t / t_r)
    v_3 *= (s / (2 * pi * t_r))
    return v_3

def test_depth_F():
    x = 1.0
    y = 0.0
    D = 10.0
    H = 10.0
    n = 3
    np.testing.assert_almost_equal(_depth_F(x, y, D, H, n),
                                   _surface_F(x, D, H, n))
