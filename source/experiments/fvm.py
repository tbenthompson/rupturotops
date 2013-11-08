import pytest
import numpy as np
from matplotlib import pyplot as pyp
from core.experiment import Experiment
from core.debug import _DEBUG
from core.update_plotter import UpdatePlotter
from core.error_tracker import ErrorTracker
from core.mesh import Mesh
assert(_DEBUG)
import experiments.wave_forms as wave_forms
import experiments.ssprk4 as ssprk4
from experiments.weno import WENO
from experiments.boundary_conds import PeriodicBC
from experiments.riemann_solver import RiemannSolver
from experiments.spatial_deriv import SpatialDeriv


class FVM(Experiment):

    def _initialize(self):
        self.t_max = 1.0
        self.temp_analytical = wave_forms.square
        self.v = 1.0
        self.delta_x = 0.01 * np.ones(100)
        if 'plotter' not in self.params:
            self.params.plotter = None
        if 'error_tracker' not in self.params:
            # Default to using the same parameters as the main plotter
            self.params.error_tracker = DataController(plotter=
                                                       self.params.plotter)
        if 't_max' in self.params:
            self.t_max = self.params.t_max
        if 'analytical' in self.params:
            self.temp_analytical = self.params.analytical
        if 'delta_x' in self.params:
            self.delta_x = self.params.delta_x

        self.mesh = Mesh(self.delta_x)
        self.delta_t = FVM.calc_time_step(self.delta_x)

        self.params.delta_t = self.delta_t

        self.analytical = lambda t: self.temp_analytical(
            (self.mesh.x - t) % self.mesh.domain_width)
        self.v = np.pad(np.ones_like(self.mesh.x), 2, 'edge')
        self.init = self.analytical(0.0)
        self.exact = self.analytical(self.t_max)

        self.riemann = RiemannSolver()
        self.bc = PeriodicBC()
        self.reconstructor = WENO(self.mesh)
        self.spatial_deriv_obj = SpatialDeriv(self.mesh, self.reconstructor,
                                  self.bc, self.riemann, self.v)

    @staticmethod
    def calc_time_step(delta_x):
        return ssprk4.cfl_max * 0.9 * delta_x

    def _compute(self):
        result = self.init.copy()
        dt = np.min(self.delta_t)
        t = dt

        self.error_tracker = ErrorTracker(self.mesh,
                                          result, self.analytical, dt,
                                          self.params.error_tracker)
        self.params.plotter.x_bounds = [self.mesh.left_edge,
                                        self.mesh.right_edge]
        self.params.plotter.y_bounds = [np.min(result), np.max(result)]
        soln_plot = UpdatePlotter(dt, self.params.plotter)

        soln_plot.add_line(self.mesh.x, result, '+')
        soln_plot.add_line(self.mesh.x, self.init, '-')

        while t <= self.t_max:
            result = ssprk4.ssprk4(self.spatial_deriv_obj.compute, result, t, dt)
            self.error_tracker.update(result, t)
            soln_plot.update(result, t)
            t += dt

        soln_plot.update(result, 0)
        soln_plot.add_line(self.mesh.x, self.exact)
        return result

    @staticmethod
    def total_variation(a):
        # add a zero padding
        return np.sum(abs(a - np.roll(a, 1)))

#----------------------------------------------------------------------------
# TESTS
#----------------------------------------------------------------------------
from core.data_controller import DataController
interactive_test = False


def test_total_variaton():
    a = [1, 0, 1, 1, 1]
    assert(FVM.total_variation(a) == 2.0)


def test_init_cond():
    params = DataController()
    params.delta_x = 0.1 * np.ones(50)
    params.analytical = wave_forms.square
    fvm = FVM(params)
    assert(fvm.init[0] == 0.0)
    assert(fvm.init[-1] == 0.0)


def test_time_step():
    delta_x = 1.0
    assert(FVM.calc_time_step(delta_x) <= ssprk4.cfl_max)


def test_mesh_initialize():
    params = DataController()
    params.delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    fvm = FVM(params)
    correct = np.array([0.25, 1.0, 1.55, 2.6])
    assert(fvm.mesh.domain_width == 3.6)
    assert(len(fvm.mesh.x) == 4)
    assert((fvm.mesh.x == correct).all())
    assert((np.min(fvm.delta_t) <= 0.1 * ssprk4.cfl_max))


def test_analytical_periodicity():
    fvm = FVM()
    np.testing.assert_almost_equal(fvm.analytical(fvm.mesh.domain_width),
                                   fvm.analytical(0.0))


def test_fvm_boundaries():
    delta_x = 0.1 * np.ones(50)
    _test_fvm_helper(wave_forms.square, 6.0, delta_x, 0.1)


def test_fvm_simple():
    delta_x = 0.005 * np.ones(200)
    _test_fvm_helper(wave_forms.sin_4, 2.0, delta_x, 0.05)

# This test shouldn't work with the current WENO implementation because
# the reconstruction is uniform in space
@pytest.mark.xfail
def test_fvm_varying_spacing():
    delta_x = []
    for i in range(100):
        if i % 2 == 0:
            delta_x.append(0.01)
        else:
            delta_x.append(0.1)
    delta_x = np.array(delta_x)
    _test_fvm_helper(lambda x: wave_forms.sin_4(x, np.pi / 2.0),
                     2.0, delta_x, 0.1)


def _test_fvm_helper(wave, t_max, delta_x, error_bound):
    # Simple test to make sure the code works right
    my_params = DataController()
    my_params.delta_x = delta_x
    my_params.plotter = DataController()
    my_params.plotter.always_plot = False
    my_params.plotter.never_plot = not interactive_test
    my_params.plotter.plot_interval = 0.5
    my_params.t_max = t_max
    my_params.analytical = wave
    fvm = FVM(my_params)
    result = fvm.compute()

    # check essentially non-oscillatoriness
    # total variation <= initial_tv + O(h^2)
    init_tv = FVM.total_variation(fvm.init)
    result_tv = FVM.total_variation(result)
    assert(result_tv < init_tv + error_bound)

    # check error
    assert(fvm.error_tracker.error[-1] < error_bound)

    if interactive_test is True:
        pyp.show()
