import numpy as np
from matplotlib import pyplot as pyp
import scipy.integrate
from utilities import calc_strain, finish_calc_params
# from debug import _DEBUG
from parameters import params
from experiment import Experiment

class ShearHeating(Experiment):
    pass

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
    finish_calc_params(params)
    data = params.copy()
    pyp.plot(data['X'], data['initial_temp'], '.')
    pyp.show()
    # exit()
    final_temp_exp = run(data)
    exp_strain = calc_strain(final_temp_exp.T, data)[:, -1]

    # pyp.plot(data['X'], final_temp_exp.T)
    # pyp.show()

    data['include_exp'] = False
    final_temp_no_exp = run(data)
    no_exp_strain = calc_strain(final_temp_no_exp.T, data)[:, -1]

    strain_diff = exp_strain - no_exp_strain
    pyp.plot(data['X'], strain_diff)
    pyp.plot(data['X'], exp_strain)
    pyp.show()
    print np.max(strain_diff)
    data['results'] = dict()
    data['results']['final_temp_exp'] = final_temp_exp
    data['results']['final_temp_no_exp'] = final_temp_no_exp
    data['results']['exp_strain'] = exp_strain
    data['results']['no_exp_strain'] = no_exp_strain
    data['results']['strain_diff'] = strain_diff

if __name__ == "__main__":
    # compare_exp_and_no_exp()
    total_strain_plot()
