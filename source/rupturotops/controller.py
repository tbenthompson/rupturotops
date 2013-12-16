import pytest
import numpy as np
from matplotlib import pyplot as pyp
from core.experiment import Experiment
from core.debug import _DEBUG
from core.update_plotter import UpdatePlotter
from core.error_tracker import ErrorTracker
from core.mesh import Mesh
assert(_DEBUG)
import rupturotops.wave_forms as wave_forms
import rupturotops.ssprk4 as ssprk4
from rupturotops.weno import WENO, WENO_NEW2
from rupturotops.boundary_conds import PeriodicBC
from rupturotops.spatial_deriv import SimpleFlux


class Controller(Experiment):

    def _initialize(self):
        self.t_max = 1.0
        self.temp_analytical = wave_forms.square
        self.v = 1.0
        self.delta_x = 0.01 * np.ones(100)
        self.handle_params()

        self.mesh = Mesh(self.delta_x)

        self.analytical = lambda t: self.temp_analytical(
            (self.mesh.x - t) % self.mesh.domain_width)
        self.v = np.pad(np.ones_like(self.mesh.x), 2, 'edge')
        self.init = self.analytical(0.0)
        self.exact = self.analytical(self.t_max)

        # The pieces of a finite volume scheme
        self.timestepper = ssprk4.SSPRK4()
        self.bc = PeriodicBC()
        self.reconstructor = WENO_NEW2(self.mesh)
        self.spatial_deriv_obj = SimpleFlux(self.reconstructor, self.v, self.mesh, self.bc)
        self.observers = []

    def handle_params(self):
        if 'plotter' not in self.params:
            self.params.plotter = None
        if 'error_tracker' not in self.params:
            # Default to using the same parameters as the main plotter
            self.params.error_tracker = Data(plotter=
                                             self.params.plotter)
        if 't_max' in self.params:
            self.t_max = self.params.t_max
        if 'analytical' in self.params:
            self.temp_analytical = self.params.analytical
        if 'delta_x' in self.params:
            self.delta_x = self.params.delta_x

    def update_observers(self, result, t, dt):
        for o in self.observers:
            o.update(result, t, dt)

    def _compute(self):
        # Initialize the solver variables.
        result = self.init.copy()

        # Assumes v is constant
        dt = self.timestepper.get_dt(self.v[0], self.mesh.delta_x)
        t = 0
        # The main program loop
        while t <= self.t_max:
            result = self.timestepper.compute(
                self.spatial_deriv_obj.compute,
                result, t, dt)
            t += dt
            # Update the plots
            self.update_observers(result, t, dt)

        # final update
        self.update_observers(result, t, dt)
        return result

    @staticmethod
    def total_variation(a):
        # add a zero padding
        return np.sum(abs(a - np.roll(a, 1)))

    def _visualize(self):
        pass

#----------------------------------------------------------------------------
# TESTS
#----------------------------------------------------------------------------
from core.data import Data
interactive_test = True


def test_total_variaton():
    a = [1, 0, 1, 1, 1]
    assert(Controller.total_variation(a) == 2.0)


def test_init_cond():
    params = Data()
    params.delta_x = 0.1 * np.ones(50)
    params.analytical = wave_forms.square
    controller = Controller(params)
    assert(controller.init[0] == 0.0)
    assert(controller.init[-1] == 0.0)


def test_mesh_initialize():
    params = Data()
    params.delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    controller = Controller(params)
    correct = np.array([0.25, 1.0, 1.55, 2.6])
    assert(controller.mesh.domain_width == 3.6)
    assert(len(controller.mesh.x) == 4)
    assert((controller.mesh.x == correct).all())


def test_analytical_periodicity():
    controller = Controller()
    np.testing.assert_almost_equal(
        controller.analytical(controller.mesh.domain_width),
        controller.analytical(0.0))


def test_controller_boundaries():
    delta_x = 0.1 * np.ones(50)
    _test_controller_helper(wave_forms.square, 6.0, delta_x, 0.35)


def test_controller_simple():
    delta_x = 0.005 * np.ones(200)
    _test_controller_helper(wave_forms.sin_4, 2.0, delta_x, 0.05)

# This test should only work the newer WENO setup because
# the reconstruction is nonuniform in space


def test_controller_varying_spacing():
    delta_x = []
    for i in range(100):
        if i % 2 == 0:
            delta_x.append(0.01)
        else:
            delta_x.append(0.1)
    delta_x = np.array(delta_x)
    width = np.sum(delta_x)
    _test_controller_helper(lambda x: wave_forms.sin_4(x, np.pi / width),
                            5.0, delta_x, 0.15)


def test_update_count():
    delta_x = np.ones(10) * 0.2
    et, sp = _test_controller_helper(wave_forms.square, 2.0, delta_x, 10.0)
    assert(et.current_plot.update_count == (3 if interactive_test else 0))
    assert(et.all_time_plot.update_count == (3 if interactive_test else 0))
    assert(sp.update_count == (3 if interactive_test else 0))


def _test_controller_helper(wave, t_max, delta_x, error_bound, always=False,
                            setup_callback=None):
    # Simple test to make sure the code works right
    my_params = Data()
    my_params.delta_x = delta_x
    my_params.plotter = Data()
    my_params.plotter.always_plot = always
    my_params.plotter.never_plot = not interactive_test
    my_params.plotter.plot_interval = 0.5
    my_params.t_max = t_max
    my_params.analytical = wave
    cont = Controller(my_params)
    if setup_callback is not None:
        setup_callback(cont)
    et = ErrorTracker(cont.mesh, cont.analytical, my_params)

    soln_plot = UpdatePlotter(my_params.plotter)

    soln_plot.add_line(cont.mesh.x, cont.init, '+')
    soln_plot.add_line(cont.mesh.x, cont.init, '-')
    cont.observers.append(soln_plot)
    cont.observers.append(et)
    result = cont.compute()
    soln_plot.add_line(cont.mesh.x, cont.exact)

    # check essentially non-oscillatoriness
    # total variation <= initial_tv + O(h^2)
    init_tv = Controller.total_variation(cont.init)
    result_tv = Controller.total_variation(result)

    if interactive_test is True:
        pyp.show()
    assert(result_tv < init_tv + error_bound)

    # check error
    assert(et.error[-1] < error_bound)
    return et, soln_plot
