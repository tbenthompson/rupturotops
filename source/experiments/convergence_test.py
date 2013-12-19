import numpy as np
from matplotlib import pyplot as pyp
from core.data import Data
import rupturotops.wave_forms as wave_forms
from rupturotops.controller import Controller
from core.debug import _DEBUG
from core.error_tracker import ErrorTracker
from core.update_plotter import UpdatePlotter

class ConvergenceTest(object):
    def __init__(self, params, domain_length):
        self.params = params
        self.domain_length = domain_length
        self.dx_exp = -np.arange(1, 7)
        self.dx = 2.0 ** self.dx_exp

    def run(self):
        errors = []
        for i in range(len(self.dx)):
            print self.dx[i]
            delta_x = np.ones(self.domain_length / self.dx[i]) * self.dx[i]
            self.params.delta_x = delta_x
            cont = Controller(self.params)
            et = ErrorTracker(cont.mesh, cont.analytical, self.params)
            #soln_plot = UpdatePlotter(self.params.plotter)
            #soln_plot.add_line(cont.mesh.x, cont.init, '+')
            #soln_plot.add_line(cont.mesh.x, cont.init, '-')
            #cont.observers.append(soln_plot)
            cont.observers.append(et)
            cont.compute()
            errors.append(et.get_final_error())
        pyp.close('all')
        pyp.figure()
        pyp.plot(np.log(self.dx), np.log(errors))
        pyp.plot([-2.0, -3.0], [-2, -7])
        pyp.xlabel(r'$\log(\Delta x)$')
        pyp.ylabel(r'log(error)')
        pyp.title('Convergence plot for ADER-WENO method')
        pyp.show()

def test_conv_test():
    my_params = Data()
    my_params.plotter = Data()
    my_params.plotter.always_plot = False
    my_params.plotter.never_plot = True
    my_params.plotter.plot_interval = 0.5
    my_params.t_max = 50.0
    #my_params.analytical = lambda x: wave_forms.sin_4(x, 2 * np.pi * 1.0)
    my_params.analytical = lambda x: wave_forms.gaussian(x, 4.0, 2.5)
    ct = ConvergenceTest(my_params, 5.0)
    # ct.run()
