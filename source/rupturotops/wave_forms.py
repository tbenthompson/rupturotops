import numpy as np

def square(x):
    return np.where(np.logical_and(x <= 1.0, x >= 0.5),
                np.ones_like(x), np.zeros_like(x))

def gaussian(x, width=1.0, center=1.0):
    return np.exp(-(x - center) ** 2 * width)

def step(x):
    return np.where(x < 1.0,
                    np.ones_like(x),
                    np.zeros_like(x))

def sin_4(x, width = 10.0):
    return np.sin(width * x) ** 4

def test_wave_forms():
    assert(step(1.5 - 0) == 0.0)
    assert(step(0.5 - 0) == 1.0)
    assert(step(2.5 - 3) == 1.0)
    assert(gaussian(1.0 - 0) == 1.0)
    assert(gaussian(0.0) == 1.0/np.exp(1))
    assert(gaussian(0.0, 2.0) == 1.0/np.exp(2))
    assert(square(0.75 - 0) == 1.0)
    assert(square(1.5 - 0.4) == 0.0)
    assert(sin_4(0 - 0) == 0.0)
    assert(sin_4(2 - 1, 3.0 * np.pi/2.0) == 1.0)
