import numpy as np
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG
from utilities import calc_strain
from parameters import params
import scipy.integrate


# boundary conditions are dirichlet and specified as the
# first and last component of initial_temp
def run(data):
    # _DEBUG()
    def rhs(current_temp, t):
        # print t
        # if data['plot_every'] is not None and \
                 # step % data['plot_every'] == 0:
        # pyp.plot(data['X'], current_temp)

        exp_term = 0
        if data['include_exp']:
            exp_term = data['the_constant'] * \
                np.exp(-1 / (current_temp[1:-1]))

        diffusion_term = (current_temp[:-2] - 2 * current_temp[1:-1]
                          + current_temp[2:]) / (data['delta_x'] ** 2)

        retval = np.zeros((len(current_temp)))
        retval[1:-1] = diffusion_term + exp_term
        return retval

    return scipy.integrate.odeint(rhs, data['initial_temp'], data['t'])


def total_strain_plot():
    data = params.copy()
    finalTemp = run(data)

    pyp.plot(data['X'], finalTemp.T)
    pyp.show()


    exp_strain = calc_strain(finalTemp.T, data)[:, -1]

    data['include_exp'] = False
    finalTemp = run(data)
    no_exp_strain = calc_strain(finalTemp.T, data)[:, -1]
    strain_diff = exp_strain - no_exp_strain
    # _DEBUG()
    pyp.plot(data['X'], strain_diff)
    pyp.plot(data['X'], exp_strain)
    pyp.show()

if __name__ == "__main__":
    # compare_exp_and_no_exp()
    total_strain_plot()
