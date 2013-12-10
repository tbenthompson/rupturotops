import numpy as np
from matplotlib import pyplot as pyp
from core.data import Data
import rupturotops.wave_forms as wave_forms
from rupturotops.controller import Controller
from core.debug import _DEBUG

class ConvergenceTest(object):
    def __init__(self, params, domain_length):
        self.params = params
        self.domain_length = domain_length
        self.dx_exp = -np.arange(1, 6)
        self.dx = 2.0 ** self.dx_exp

    def run(self):
        errors = []
        for i in range(len(self.dx)):
            print self.dx[i]
            delta_x = np.ones(self.domain_length / self.dx[i]) * self.dx[i]
            self.params.delta_x = delta_x
            cont = Controller(self.params)
            cont.compute()
            errors.append(cont.error_tracker.get_final_error())
        # pyp.close('all')
        # pyp.figure()
        # pyp.plot(np.log(self.dx), np.log(errors))
        # pyp.show()

def test_conv_test():
    my_params = Data()
    my_params.plotter = Data()
    my_params.plotter.always_plot = False
    my_params.plotter.never_plot = True
    my_params.plotter.plot_interval = 0.5
    my_params.t_max = 50.0
    # my_params.analytical = lambda x: wave_forms.sin_4(x, 2 * np.pi * 4.0)
    my_params.analytical = lambda x: wave_forms.gaussian(x, 4.0, 2.5)
    ct = ConvergenceTest(my_params, 5.0)
    # ct.run()
