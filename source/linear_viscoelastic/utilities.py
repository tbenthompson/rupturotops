import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG

def mse(x, y):
    return np.sqrt(np.sum((np.array(x)-np.array(y))**2)/len(x))

def dimensionless_arctan(x, alpha, C):
    retval = np.arctan((1/((alpha**2) * x)) + (float(C)/x))
    # print retval
    return retval

def A_integral(slip_distribution, x, alpha, C):
    def integrand(z, x_in):
        denom = ((alpha * x_in) ** 2 + ((z / alpha) + C * alpha) ** 2)
        return slip_distribution(z) / denom
    def integration_controller(x_in):
        return integrate.quad(integrand, 0, 1, args = (x_in,))[0]
    #don't input x = 0, singularity alert!
    return np.array(map(integration_controller, x))

def plot_time_series_1D(x, u, t, show = True):
    expand_graph_factor = 1.2
    lines = []
    for i in range(len(t)):
        lines.append(pyp.plot(x, u[i])[0])
    pyp.legend(lines, map(lambda x: "time = " + str(int(x)), t))
    #find the edges of the desired graphing area
    #don't allow positive left edges or negative right edges
    axis_list = np.array([np.min([0, np.min(x)]), np.max([np.max(x), 0]), np.min([0, np.min(np.min(u))]), np.max([0, np.max(np.max(u))])])
    pyp.axis(axis_list * expand_graph_factor)
    pyp.xlabel("Distance from the fault")
    pyp.ylabel("displacement")
    if show:
        pyp.show()
