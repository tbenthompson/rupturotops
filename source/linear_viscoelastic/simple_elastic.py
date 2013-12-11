import numpy as np
from core.debug import _DEBUG
from matplotlib import pyplot as pyp


def elastic_stress(x, y, s, D, shear_modulus):
    """
    Use the elastic half-space stress solution from Segall (2010)
    """
    # TEST THIS!
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = -(y - D) / ((y - D) ** 2 + x ** 2)
    image_term = (y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = x / (x ** 2 + (y - D) ** 2)
    image_term = -x / (x ** 2 + (y + D) ** 2)
    Szy = factor * (main_term + image_term)
    return Szx, Szy


def calc_divergence(f, g, dx, dy):
    """
    Calculate the discrete divergence of a 2D vector field.
    This should be moved to some other location.
    """
    dfdx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dx)
    dgdy = (np.roll(g, -1, 0) - np.roll(g, 1, 0)) / (2 * dy)
    return (dfdx + dgdy)[1:-1, 1:-1]


def calc_curl(f, dx, dy):
    """
    Calculates the discrete curl of a vector field with no component
    in x and y.
    """
    dfdx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * dx)
    dfdy = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * dy)
    return dfdy[1:-1, 1:-1], -dfdx[1:-1, 1:-1]


def test_div_curl():
    # The curl of anything should be divergence free
    z = np.random.random_sample((50, 50))
    curl = calc_curl(z, 1.0, 1.0)
    div = calc_divergence(curl[0], curl[1], 1.0, 1.0)
    assert(np.mean(np.abs(div)) < np.mean(np.abs(curl[0])) / 1000.0)


def test_elastic_stress():
    # Here, I just test that its divergence free.
    # Incomplete test, but better than nothing.
    x = np.linspace(1, 10, 40)
    y = np.linspace(0, 9, 40)
    x, y = np.meshgrid(x, y)
    s = 1.0
    D = 10.0
    mu = 1000000
    f = elastic_stress(x, y, s, D, mu)
    divf = calc_divergence(f[0], f[1], 9.0 / 40.0, 9.0 / 40.0)
    assert(np.mean(np.abs(divf)) < np.mean(np.abs(f[0])) / 1000.0)
