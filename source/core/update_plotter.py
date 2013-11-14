from matplotlib import pyplot as pyp
from core.data import Data
from core.debug import _DEBUG
assert(Data)


class UpdatePlotter(object):

    """
    Plot something every time step using this class

    Requires the parameters:
        always_plot -- ignore the time interval
        plot_interval -- time interval between plots
        never_plot -- never plot!
    """

    def __init__(self, params=None):
        # Defaults
        self.always_plot = True
        self.plot_interval = -1
        self.never_plot = False
        self.autoscale = True
        self.x_bounds = None
        self.y_bounds = None

        if params is not None:
            self.handle_params(params)

        self.update_count = 0
        self.since_last_update = 0.0

        self.fig = pyp.figure()
        self.ax = self.fig.add_subplot(111)
        if self.x_bounds is not None:
            self.ax.set_xbound(*self.x_bounds)
        if self.y_bounds is not None:
            self.ax.set_ybound(*self.y_bounds)
        if self.never_plot is not True:
            self.fig.show()
        self.lines = []
        self.first_plot = True

    def handle_params(self, params):
        if 'always_plot' in params:
            self.always_plot = params.always_plot
        if 'plot_interval' in params:
            self.plot_interval = params.plot_interval
        if 'never_plot' in params:
            self.never_plot = params.never_plot
        if 'autoscale' in params:
            self.autoscale = params.autoscale
        if 'x_bounds' in params:
            self.x_bounds = params.x_bounds
        if 'y_bounds' in params:
            self.y_bounds = params.y_bounds

    def add_line(self, x, y, plot_args=[]):
        self.lines.append(self.ax.plot(x, y, *plot_args)[0])
        return len(self.lines) - 1

    def init_plots(self):
        pass

    def update(self, y, t, dt, which_line=0, x=None):
        self.since_last_update += dt
        if self.first_plot is True:
            self.init_plots()
            self.first_plot = False
        if self.never_plot:
            return
        if (not self.always_plot) and \
                self.since_last_update < self.plot_interval:
            return
        if x is not None:
            self.lines[which_line].set_xdata(x)
        self.lines[which_line].set_ydata(y)
        if self.autoscale:
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.update_count += 1
        self.since_last_update = self.since_last_update - self.plot_interval

#-----------------------------------------------------------------------------
# TESTS --
#-----------------------------------------------------------------------------
import numpy as np
# it's hard to test this non-interactively. Set this to true to check it!
interactive_test = False


def test_2_plotters():
    pyp.close('all')
    x = np.linspace(0, 1, 100)
    y = 1 - (x ** 2)
    y2 = x ** 3

    param = Data(never_plot=False)
    plotter = UpdatePlotter(param)
    plotter.add_line(x, y)
    plotter.update(y2, 0, 0.1)
    assert(plotter.lines[0].get_ydata()[50] == x[50] ** 3)

    plotter2 = UpdatePlotter(param)
    plotter2.add_line(x, y2, plot_args=['*'])
    plotter2.add_line(x, np.zeros_like(x))
    plotter2.update(y, 1, 0.1)
    assert(plotter2.update_count == 1)
    assert(plotter2.lines[0].get_ydata()[50] == 1 - x[50] ** 2)

    assert(len(pyp.get_fignums()) == 2)
    if interactive_test is True:
        pyp.show()
