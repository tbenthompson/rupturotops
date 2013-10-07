import numpy as np
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG
from wetdiabase import wetdiabase
import scipy.integrate


# boundary conditions are dirichlet and specified as the
# first and last component of initial_temp
def run(initial_temp, P, steps, delta_t, delta_x, include_exp=True, plot_every=None):
    # _DEBUG()
    X = np.linspace(0, 1, len(initial_temp))
    T = np.tile(initial_temp, (1, steps))
    if plot_every is not None:
        pyp.plot(X, T[:, 0])
    for which_step in range(1, steps):
        current_temp = T[:, which_step - 1]

        exp_term = 0
        if include_exp:
            exp_term = P * np.exp(-1 / (current_temp[1:-1]))
        # print exp_term

        #centered space
        diffusion_term = (current_temp[:-2] - 2 * current_temp[1:-1]
                          + current_temp[2:]) / (delta_x ** 2)

        #forward time
        T[1:-1, which_step] = current_temp[1:-1] + \
            delta_t * (diffusion_term + exp_term)

        if plot_every is not None and which_step % plot_every == 0:
            pyp.plot(X, T[:, which_step])
    return T


def calc_strain(temp, stress, time_scale, deltat, rock_params):
    tointegrate = rock_params['creepconstant'] * time_scale * stress ** rock_params['stressexponent'] * np.exp(-1 / temp)
    strain = scipy.integrate.cumtrapz(tointegrate, dx=deltat)
    return strain


def find_consts(stress, length_scale, rock_params):
    temp_scale = rock_params['activationenergy'] / rock_params['R']
    time_scale = length_scale ** 2 / rock_params['thermaldiffusivity']
    numer = (time_scale * rock_params['creepconstant'] * stress ** (rock_params['stressexponent'] + 1))
    denom = (temp_scale * rock_params['specificheat'] * rock_params['density'])
    return temp_scale, time_scale, numer / denom


def compare_exp_and_no_exp():
    stress = 100e6
    length_scale = 200
    temp_scale, time_scale, P = find_consts(stress, length_scale, wetdiabase)

    #algorithm parameters
    steps = 500000
    plot_steps = 25000
    delta_t = 0.00001
    delta_x = 0.05
    points = 50

    #initial conditions and boundary conditions (in the first and last element
    # of the array)
    initialTemp = 800 / temp_scale
    temp = np.ones((points + 1, 1)) * initialTemp
    temp[points / 2 + 1] = 1200 / temp_scale

    run(temp, P, steps, delta_t, delta_x, include_exp=False, plot_every=plot_steps)
    pyp.axis([0, 1, initialTemp, initialTemp * 1.3])
    # pyp.show()
    pyp.figure()
    run(temp, P, steps, delta_t, delta_x, include_exp=True, plot_every=plot_steps)
    pyp.axis([0, 1, initialTemp, initialTemp * 1.3])
    pyp.show()


def total_strain_plot():
    stress = 100e6
    length_scale = 20
    temp_scale, time_scale, P = find_consts(stress, length_scale, wetdiabase)

    #algorithm parameters
    steps = 50000
    plot_steps = 5000
    delta_t = 0.00001
    delta_x = 0.05
    points = 50

    #initial conditions and boundary conditions (in the first and last element
    #of the array)
    initialTemp = 800 / temp_scale
    temp = np.ones((points + 1, 1)) * initialTemp
    temp[points / 2] = 1100 / temp_scale
    temp[points / 2 + 1] = 1400 / temp_scale
    temp[points / 2 + 2] = 1100 / temp_scale

    finalTemp = run(temp, P, steps, delta_t, delta_x, include_exp=True,
                    plot_every=plot_steps)
    strain = calc_strain(finalTemp, stress, time_scale, delta_t, wetdiabase)
    exp_strain = strain[:, -1]
    # pyp.plot(np.linspace(0, 1, points + 1), final_strain)
    # pyp.axis([0,1, 1e-8, 1e-7])

    finalTemp = run(temp, P, steps, delta_t, delta_x, include_exp=False,
                    plot_every=plot_steps)
    pyp.show()
    strain = calc_strain(finalTemp, stress, time_scale, delta_t, wetdiabase)
    no_exp_strain = strain[:, -1]
    strain_diff = exp_strain - no_exp_strain
    pyp.plot(np.linspace(0, 1, points + 1), strain_diff)
    pyp.show()
    _DEBUG()

if __name__ == "__main__":
    # compare_exp_and_no_exp()
    total_strain_plot()
