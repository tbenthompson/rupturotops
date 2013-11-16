
class SSPRK4(object):
    cfl_max = 1.507

    def __init__(self):
        pass

    def get_dt(self, v, dx, factor=0.9):
        """
        Find the proper time step for the Runge Kutta scheme. By default
        takes a time-step 9/10ths of the cfl maximum.
        """
        dt = factor * (self.cfl_max * dx) / v
        return np.min(dt)

    def compute(self, f, current, t, dt):
        """
        Implements the linear, 4th order, 5 stage Strong Stability
        Preserving Runge Kutta scheme from Gottlieb, Ketcheson, Shu (2009).
        This is the nonlinear one, I should change this to use the linear
        case. It has a higher CFL-max of 2 vs. the current 1.507
        """
        t1 = current + 0.391752226571890 * dt * f(t, current)
        t2 = 0.444370493651235 * current + \
            0.555629506348765 * t1 + \
            0.368410593050371 * dt * f(t, t1)
        t3 = 0.620101851488403 * current + \
            0.379898148511597 * t2 + \
            0.251891774271694 * dt * f(t, t2)
        f3 = f(t, t3)
        t4 = 0.178079954393132 * current + \
            0.821920045606868 * t3 + \
            0.544974750228521 * dt * f3
        final = 0.517231671970585 * t2 + \
            0.096059710526147 * t3 + \
            0.063692468666290 * dt * f3 + \
            0.386708617503269 * t4 + \
            0.226007483236906 * dt * f(t, t4)
        return final

#------------------------------------------
# TESTS
#------------------------------------------
import numpy as np


def test_ssprk4():
    # Really simple test just to make sure it's reasonable
    f = lambda t, x: 1
    x = np.linspace(0, 5, 10)
    dt = 1
    s = SSPRK4()
    result = s.compute(f, x, 0, dt)
    diff = abs(result - (x + 1))
    np.testing.assert_almost_equal(diff, 0.0)

def test_get_dt():
    s = SSPRK4()
    v = 10.0
    dx = 0.1
    np.testing.assert_almost_equal(s.get_dt(v, dx), 0.009 * s.cfl_max)
