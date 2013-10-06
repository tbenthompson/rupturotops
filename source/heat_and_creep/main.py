import numpy as np
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG


# boundary conditions are dirichlet and specified as the
# first and last component of initial_temp
def run(initial_temp, P, steps, delta_t, delta_x, include_exp=True, plot_every=None):
    # _DEBUG()
    X = np.linspace(0, 1, len(initial_temp))
    T = np.tile(initial_temp, (1, steps))
    pyp.plot(X, T[:, 0])
    for which_step in range(1, steps):
        current_temp = T[:, which_step - 1]

        exp_term = 0
        if include_exp:
            exp_term = P * np.exp(-1 / current_temp[1:-1])
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


if __name__ == "__main__":
    temp = np.ones((201, 1))
    temp[100] = 2.0
    P = 0.0001
    steps = 500
    delta_t = 1.0
    delta_x = 2.0
    run(temp, P, steps, delta_t, delta_x, include_exp=False, plot_every=50)
    # pyp.axis([0, 1, 1, 1.2])
    # pyp.show()
    pyp.figure()
    run(temp, P, steps, delta_t, delta_x, include_exp=True, plot_every=50)
    # pyp.axis([0, 1, 1, 1.2])
    pyp.show()
