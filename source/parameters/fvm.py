from matplotlib import pyplot as pyp
from rupturotops.weno import WENO
from core.data import Data
from core.update_plotter import UpdatePlotter
from core.error_tracker import ErrorTracker
from core.experiment import Experiment
from rupturotops.controller import Controller
from parameters.material import wetdiabase
from core.constants import consts
from rupturotops import wave_forms
assert(consts)
from core.debug import _DEBUG
assert(_DEBUG)
import numpy as np

# Define the standard parameter data structure.
params = Data()

# What material should we use? Look in the parameters/material file
# to see the options and to see how to define a new material.
params.material = wetdiabase


# Setup the solution domain, first we define the cell spacings
delta_x = []
count = 2000
domain_length = 10.0
for i in range(count):
    if i % 10 == 0:
        delta_x.append(domain_length / count)
    else:
        delta_x.append(domain_length / count)
width = np.sum(delta_x)
params.delta_x = np.array(delta_x)

# plotting parameters
params.plotter = Data()
params.plotter.always_plot = False
params.plotter.never_plot = False
params.plotter.plot_interval = 0.1


params.t_max = 100.0
wavelengths = 3
# params.analytical = lambda x: wave_forms.sin_4(x, wavelengths * np.pi / width)
params.analytical = wave_forms.square

# Define project and run parameters in order to save the results to
# the proper data folder.
params.proj_name = 'fvm'
params.run_name = 'development'

# Define the set of experiments to be run
class Play(Experiment):
    def _initialize(self):
        self.cont = Controller(self.params)
        # self.cont.deriv.reconstructor = WENO(self.cont.mesh)
        et = ErrorTracker(self.cont.mesh, self.cont.analytical, self.params)
        soln_plot = UpdatePlotter(self.params.plotter)

        soln_plot.add_line(self.cont.mesh.x, self.cont.init[2:-2], '+')
        soln_plot.add_line(self.cont.mesh.x, self.cont.init[2:-2], '-')
        self.cont.observers.append(soln_plot)
        self.cont.observers.append(et)

    def _compute(self):
        self.cont._compute()

    def _visualize(self):
        pyp.show()
        self.cont._visualize()

experiment = Play
