import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as pyp

def mse(x, y):
    return np.sqrt(np.sum((np.array(x)-np.array(y))**2)/len(x))

def dimensionless_arctan(x, alpha, C):
    retval = np.arctan((1/((alpha**2) * x)) + (float(C)/x))
    # print retval
    return retval

def A_integral(slip_distribution, alpha, x, C):
    def integrand(z):
        denom = ((alpha * x) ** 2 + ((z / alpha) + C * alpha) ** 2)
        return slip_distribution(z) / denom
    return integrate.quad(integrand, 0, 1)[0]

def plot_time_series_1D(x, u, t):
    expand_graph_factor = 1.2
    lines = []
    for i in range(len(t)):
        lines.append(pyp.plot(x, u[i])[0])
    pyp.legend(lines, map(lambda x: "time = " + str(int(x)), t))
    axis_list = np.array([min(x),max(x), min(min(u)), max(max(u))])
    pyp.axis(axis_list * expand_graph_factor)
    pyp.xlabel("distance from the fault")
    pyp.ylabel("displacement")
    pyp.show()



