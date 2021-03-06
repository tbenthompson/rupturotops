import numpy as np
from core.data import Data
from core.debug import _DEBUG
assert(_DEBUG)
from core.update_plotter import UpdatePlotter


class ErrorTracker(object):

    """
    Error analysis and tracking tools.
    The "exact" parameter to the constructor should be a function
    that accepts one parameter: time. and produces the exact result.
    """

    def __init__(self, mesh, exact, params):
        # Defaults
        self.error_norm = 1
        self.params_plotter = None
        self.mesh = mesh

        if params is not None:
            self.handle_params(params)

        self.exact_soln = exact
        self.error = [0.0]

        self.current_plot = UpdatePlotter(self.params_plotter)
        self.current_plot.add_line(self.mesh.x, np.zeros_like(self.mesh.x), '*')
        self.all_time_plot = UpdatePlotter(self.params_plotter)
        self.all_time_plot.add_line([0], [0], '-')

    def handle_params(self, params):
        if 'error_norm' in params:
            self.error_norm = params.error_norm
        if 'plotter' in params:
            self.params_plotter = params.plotter

    @staticmethod
    def calc_norm(vector, norm):
        if norm is np.Inf:
            return abs(np.max(vector))
        err = np.sum(abs(vector) ** norm) ** \
            (1.0 / norm)
        return err

    def get_final_error(self):
        return self.error[-1]

    def update(self, y, t, dt):
        exact = self.exact_soln(t)
        diff = y - exact
        e = ErrorTracker.calc_norm(diff * self.mesh.delta_x, self.error_norm)
        self.error.append(e)

        self.current_plot.update(diff, t, dt)
        self.all_time_plot.update(self.error, t, dt,
                                  x=np.arange(0, len(self.error)))

#-------------------------------------------------------------------
# TESTS
#-------------------------------------------------------------------
from core.mesh import Mesh


def test_error_norm():
    vec = np.linspace(0.0, 1.0, 3.0)
    l1 = ErrorTracker.calc_norm(vec, 1)
    l2 = ErrorTracker.calc_norm(vec, 2)
    linf = ErrorTracker.calc_norm(vec, np.Inf)
    assert(l1 == 1.5)
    assert(abs(l2 - np.sqrt(1.25)) <= 1e-9)
    assert(linf == 1.0)

def test_error_tracker():
    delta_x = 0.01 * np.ones(100)
    exact = lambda t: np.linspace(0, 1, 100) + t
    params = Data()
    params.error_norm = 1
    params.plotter = Data()
    params.plotter.never_plot = True
    e1 = ErrorTracker(Mesh(delta_x), exact, params)
    params.error_norm = 2
    e2 = ErrorTracker(Mesh(delta_x), exact, params)
    for i in range(1, 10):
        # the exact result is x + i
        e1.update(e1.mesh.x + 0.9 * i, float(i), 1.0)
        # in any norm the error should be 0.i
        assert(abs(e1.error[i] - i * 0.1) <= 1e-9)

        new_val = exact(i)
        new_val[0] = 0
        # the difference
        e2.update(new_val, float(i), 1.0)
        # L2 norm of the error should be 0.(i + 1)
        assert(abs(e2.error[i] - (i * 0.01)) <= 1e-9)
