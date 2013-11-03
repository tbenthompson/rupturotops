import numpy as np
from matplotlib import pyplot as pyp
from core.experiment import Experiment
from core.debug import _DEBUG
from core.update_plotter import UpdatePlotter
from core.error_tracker import ErrorTracker
assert(_DEBUG)
from experiments.weno import WENO
import experiments.wave_forms as wave_forms
import experiments.ssprk4 as ssprk4

#All the static methods probably should be moved to their
#own classes to enforce some separation of responsibilities
#delta_t calculation, grid calculation go in meshing class
#separate boundary conditions

class FVM(Experiment):
    def _initialize(self):
        self.t_max = 1.0
        self.temp_analytical = wave_forms.square
        self.v = 1.0
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

        self.delta_x = self.params.delta_x
        self.domain_width = np.sum(self.delta_x)
        self.x = FVM.positions_from_spacing(self.delta_x)
        self.delta_t = FVM.calc_time_step(self.delta_x)

        self.params.delta_t = self.delta_t

        self.analytical = lambda t: self.temp_analytical(
            (self.x - t) % self.domain_width)
        self.v = np.pad(np.ones_like(self.x), 2, 'edge')
        self.init = self.analytical(0.0)
        self.exact = self.analytical(self.t_max)

        self.flux_control = WENO()

    @staticmethod
    def positions_from_spacing(spacings):
        """
        Sum the spacings to get cell centers,
        but the cell center is at the halfway point, so we have to
        subtract half of delta_x again.
        """
        x = np.cumsum(spacings)
        x = x - spacings / 2.0
        return x

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

    @staticmethod
    def split_velocity(v):
        rightwards = np.where(v > 0, v, 0)
        leftwards = np.roll(-np.where(v < 0, v, 0), -1)
        return rightwards, leftwards

    # @autojit()
    def spatial_deriv(self, t, now):
        now = FVM.boundary_cond(t, now)

        rightwards_v, leftwards_v = FVM.split_velocity(self.v)
        left_edge_u = leftwards_v * self.flux_control.compute(now, -1)
        right_edge_u = rightwards_v * self.flux_control.compute(now, 1)
        # Because we are solving a Riemann problem, the zeroth
        # order contribution to the flux should be the difference
        # in the left and right values at a boundary
        # outflow comes from the right side and inflow from the left
        left_side_of_left_bnd = np.roll(right_edge_u, 1)
        right_side_of_left_bnd = left_edge_u
        total_left_flux = -(right_side_of_left_bnd - left_side_of_left_bnd)
        # The total flux should be the flux coming in from the right
        # minus the flux going out the left.
        in_from_right = total_left_flux
        out_to_the_left = np.roll(total_left_flux, -1)
        total_flux = in_from_right - out_to_the_left
        return total_flux[2:-2] / self.delta_x

    def _compute(self):
        result = self.init.copy()
        dt = np.min(self.delta_t)
        t = dt

        error_tracker = ErrorTracker(self.x, result,
                                     self.analytical, dt,
                                     self.params.error_tracker)
        self.params.plotter.x_bounds = [np.min(self.x), np.max(self.x)]
        self.params.plotter.y_bounds = [np.min(result), np.max(result)]
        soln_plot = UpdatePlotter(dt, self.params.plotter)

        soln_plot.add_line(self.x, result, '+')
        soln_plot.add_line(self.x, self.init, '-')

        while t <= self.t_max:
            result = ssprk4.ssprk4(self.spatial_deriv, result, t, dt)
            t += dt
            error_tracker.update(result, t)
            soln_plot.update(result, t)

        soln_plot.update(result, 0)
        soln_plot.add_line(self.x, self.exact)
        return result

#-----------------------------------------------------------------------------
# TESTS
#-----------------------------------------------------------------------------
from core.data_controller import DataController
interactive_test = True

def test_init_cond():
    params = DataController()
    params.delta_x = 0.1 * np.ones(50)
    params.analytical = wave_forms.square
    fvm = FVM(params)
    assert(fvm.init[0] == 0.0)
    assert(fvm.init[-1] == 0.0)

def test_split_vel():
    v = np.array([1, -1, 1])
    right, left = FVM.split_velocity(v)
    assert((left == [1, 0, 0]).all())
    assert((right == [1, 0, 1]).all())

def test_time_step():
    delta_x = 1.0
    assert(FVM.calc_time_step(delta_x) <= ssprk4.cfl_max)

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

def test_circular_boundary_conditions():
    params = DataController()
    params.delta_x = 0.1 * np.ones(50)
    params.analytical = lambda x: x
    fvm = FVM(params)
    added_ghosts = fvm.boundary_cond(0, fvm.init)
    np.testing.assert_almost_equal(added_ghosts[0], 4.85)
    np.testing.assert_almost_equal(added_ghosts[1], 4.95)
    np.testing.assert_almost_equal(added_ghosts[-2], 0.05)
    np.testing.assert_almost_equal(added_ghosts[-1], 0.15)
    np.testing.assert_almost_equal(fvm.analytical(5.0), fvm.analytical(0.0))

def test_fvm_boundaries():
    _test_fvm_helper(wave_forms.square, 6.0)

def test_fvm_simple():
    _test_fvm_helper(wave_forms.sin_wave, 1.0)

def _test_fvm_helper(wave, t_max):
    #Simple test to make sure the code works right
    my_params = DataController()
    my_params.delta_x = 0.01 * np.ones(100)
    my_params.plotter = DataController()
    my_params.plotter.always_plot = False
    my_params.plotter.plot_interval = 0.5
    my_params.t_max = t_max
    my_params.analytical = wave
    fvm = FVM(my_params)
    initial = np.pad(fvm.init, 1, 'constant')
    #check essentially non-oscillatoriness
    #total variation <= initial_tv + O(h^2)
    tv = np.sum(abs(initial - np.roll(initial, 1)))
    result = np.pad(fvm.compute(), 1, 'constant')
    result_tv = np.sum(abs(result - np.roll(result, 1)))
    print result_tv - tv
    assert(result_tv < tv + 0.005)
    if interactive_test is True:
        pyp.show()
