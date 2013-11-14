from matplotlib import pyplot as pyp
from core.data import Data
assert(Data)


class UpdatePlotter(object):

    """
    Plot something every time step using this class

    Requires the parameters:
        always_plot -- ignore the time interval
        plot_interval -- time interval between plots
        never_plot -- never plot!
    """

    def __init__(self, delta_t, params=None):
        # Defaults
        self.always_plot = True
        self.plot_interval = -1
        self.never_plot = False
        self.autoscale = True
        self.x_bounds = None
        self.y_bounds = None
        self.delta_t = delta_t

        if params is not None:
            self.handle_params(params)

        self.fig = pyp.figure()
        self.ax = self.fig.add_subplot(111)
        if self.x_bounds is not None:
            self.ax.set_xbound(*self.x_bounds)
        if self.y_bounds is not None:
            self.ax.set_ybound(*self.y_bounds)
        if self.never_plot is not True:
            self.fig.show()
        self.lines = []

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

    def update(self, y, t, which_line=0, x=None):
        do_we_plot = t / self.plot_interval
        if self.never_plot:
            return
        if (not self.always_plot) and \
                abs(do_we_plot - round(do_we_plot)) > (self.delta_t / self.plot_interval):
            return
        if x is not None:
            self.lines[which_line].set_xdata(x)
        self.lines[which_line].set_ydata(y)
        if self.autoscale:
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()

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

    plotter = UpdatePlotter(0.1, None)
    plotter.add_line(x, y)
    plotter.update(y2, 0)
    assert(plotter.lines[0].get_ydata()[50] == x[50] ** 3)

    plotter2 = UpdatePlotter(0.1, Data())
    plotter2.add_line(x, y2, plot_args=['*'])
    plotter2.add_line(x, np.zeros_like(x))
    plotter2.update(y, 1)
    assert(plotter2.lines[0].get_ydata()[50] == 1 - x[50] ** 2)

    assert(len(pyp.get_fignums()) == 2)
    if interactive_test is True:
        pyp.show()
