import numpy as np
from matplotlib import pyplot as pyp
from core.experiment import Experiment
from core.debug import _DEBUG
from core.update_plotter import UpdatePlotter
from core.error_tracker import ErrorTracker
assert(_DEBUG)
from experiments.weno import WENO, WENO_NEW
import experiments.wave_forms as wave_forms
import experiments.ssprk4 as ssprk4
from experiments.riemann_solver import RiemannSolver

#All the static methods probably should be moved to their
#own classes to enforce some separation of responsibilities
#delta_t calculation, grid calculation go in meshing class
#separate boundary conditions

class FVM(Experiment):
    def _initialize(self):
        self.t_max = 1.0
        self.temp_analytical = wave_forms.square
        self.v = 1.0
        self.delta_x = 0.01 * np.ones(100)
        if 'plotter' not in self.params:
            self.params.plotter = None
        if 'error_tracker' not in self.params:
            #Default to using the same parameters as the main plotter
            self.params.error_tracker = DataController(plotter=
                                                       self.params.plotter)
        if 't_max' in self.params:
            self.t_max = self.params.t_max
        if 'analytical' in self.params:
            self.temp_analytical = self.params.analytical
        if 'delta_x' in self.params:
            self.delta_x = self.params.delta_x

        self.x, self.domain_width = FVM.positions_from_spacing(self.delta_x)
        self.delta_t = FVM.calc_time_step(self.delta_x)

        self.params.delta_t = self.delta_t

        self.analytical = lambda t: self.temp_analytical(
            (self.x - t) % self.domain_width)
        self.v = np.pad(np.ones_like(self.x), 2, 'edge')
        self.init = self.analytical(0.0)
        self.exact = self.analytical(self.t_max)

        self.riemann = RiemannSolver()
        self.reconstructor = WENO_NEW(7)

    @staticmethod
    def positions_from_spacing(spacings):
        """
        Sum the spacings to get cell centers,
        but the cell center is at the halfway point, so we have to
        subtract half of delta_x again.
        """
        x = np.cumsum(spacings)
        domain_width = x[-1]
        x = x - spacings / 2.0
        return x, domain_width

    @staticmethod
    def calc_time_step(delta_x):
        return ssprk4.cfl_max * 0.9 * delta_x

    @staticmethod
    def boundary_cond(t, now):
        ghosts = np.pad(now, 2, 'constant')
        ghosts[0] = ghosts[-4]
        ghosts[1] = ghosts[-3]  # np.sin(30 * t)
        ghosts[-2] = ghosts[2]
        ghosts[-1] = ghosts[3]
        return ghosts

    # @autojit()
    def spatial_deriv(self, t, now):
        now = FVM.boundary_cond(t, now)
        recon_left = self.reconstructor.compute(now, 1)
        recon_right = self.reconstructor.compute(now, -1)

        rightwards_flux = self.riemann.compute(recon_left,
                                               recon_right,
                                               self.v)
        # The total flux should be the flux coming in from the right
        # minus the flux going out the left.
        leftwards_flux = -np.roll(rightwards_flux, -1)
        total_flux = rightwards_flux + leftwards_flux
        return total_flux[2:-2] / self.delta_x

    def _compute(self):
        result = self.init.copy()
        dt = np.min(self.delta_t)
        t = dt

        self.error_tracker = ErrorTracker(self.x, self.delta_x,
                                          result, self.analytical, dt,
                                     self.params.error_tracker)
        self.params.plotter.x_bounds = [np.min(self.x), np.max(self.x)]
        self.params.plotter.y_bounds = [np.min(result), np.max(result)]
        soln_plot = UpdatePlotter(dt, self.params.plotter)

        soln_plot.add_line(self.x, result, '+')
        soln_plot.add_line(self.x, self.init, '-')

        while t <= self.t_max:
            result = ssprk4.ssprk4(self.spatial_deriv, result, t, dt)
            self.error_tracker.update(result, t)
            soln_plot.update(result, t)
            t += dt

        soln_plot.update(result, 0)
        soln_plot.add_line(self.x, self.exact)
        return result

    @staticmethod
    def total_variation(a):
        #add a zero padding
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

def test_positions_from_spacing():
    delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    correct = np.array([0.25, 1.0, 1.55, 2.6])
    x, width = FVM.positions_from_spacing(delta_x)
    assert((x == correct).all())
    assert(width == 3.6)

def test_mesh_initialize():
    params = DataController()
    params.delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    fvm = FVM(params)
    correct = np.array([0.25, 1.0, 1.55, 2.6])
    assert(fvm.domain_width == 3.6)
    assert(len(fvm.x) == 4)
    assert((fvm.x == correct).all())
    assert((np.min(fvm.delta_t) <= 0.1 * ssprk4.cfl_max))

def test_fvm_flux_compute():
    params = DataController()
    params.delta_x = 0.1 * np.ones(50)
    params.analytical = wave_forms.square
    fvm = FVM(params)
    deriv = fvm.spatial_deriv(0, fvm.init)
    assert(np.sum(deriv) <= 0.005)
    for i in range(0, len(deriv)):
        if i == 5:
            np.testing.assert_almost_equal(deriv[5], -10.0)
            continue
        if i == 10:
            np.testing.assert_almost_equal(deriv[10], 10.0)
            continue
        np.testing.assert_almost_equal(deriv[i], 0.0)

def test_analytical_periodicity():
    fvm = FVM()
    np.testing.assert_almost_equal(fvm.analytical(fvm.domain_width), fvm.analytical(0.0))

def test_fvm_boundaries():
    delta_x = 0.1 * np.ones(50)
    _test_fvm_helper(wave_forms.square, 6.0, delta_x, 0.1)

def test_fvm_simple():
    delta_x = 0.005 * np.ones(1000)
    _test_fvm_helper(wave_forms.sin_4, 1.0, delta_x, 0.05)

def test_fvm_varying_spacing():
    delta_x = np.linspace(0.02, 0.02, 800)
    _test_fvm_helper(lambda x: wave_forms.sin_4(x, np.pi/2.0),
                     2.0, delta_x, 0.1)

def _test_fvm_helper(wave, t_max, delta_x, error_bound):
    #Simple test to make sure the code works right
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

    #check essentially non-oscillatoriness
    #total variation <= initial_tv + O(h^2)
    init_tv = FVM.total_variation(fvm.init)
    result_tv = FVM.total_variation(result)
    assert(result_tv < init_tv + error_bound)

    #check error
    assert(fvm.error_tracker.error[-1] < error_bound)

    if interactive_test is True:
        pyp.show()
