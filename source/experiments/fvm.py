import numpy as np
from matplotlib import pyplot as pyp
import time
from core.debug import _DEBUG
from numba import autojit
# import scipy.io
from weno import WENO


class FVM(object):

    def __init__(self, which_init):
        self.x_elements = 300
        self.x_min = 0.0
        self.x_max = 5.0
        self.delta_x = (self.x_max - self.x_min) / (self.x_elements - 1)
        self.delta_t = 1.5 * self.delta_x
        courant = self.delta_t / self.delta_x
        print('Courant number: ' + str(courant))
        # periods = 2.0
        # t_max = 4.0 * periods
        self.t_max = 30.0
        self.plot = True
        self.always_plot = True
        self.plot_interval = 5.0
        self.x = np.linspace(self.x_min, self.x_max, self.x_elements)
        self.v = -np.pad(np.ones_like(self.x), 2, 'edge')
        if which_init is 'square':
            self.analytical = lambda t: np.where(
                np.logical_and(self.x < (1.0 + t), self.x > (0.5 + t)),
                np.ones_like(self.x), np.zeros_like(self.x))
        elif which_init is 'smooth':
            self.gaussian_width = 5
            self.analytical = lambda t: np.exp(-(self.x - t - 1.0) ** 2 * self.gaussian_width)
        elif which_init is 'step':
            self.analytical = lambda t: np.where(
                self.x < (1.0 + t), np.ones_like(self.x), np.zeros_like(self.x))
        elif which_init is 'jumpy':
            self.analytical = lambda t: np.sin(10 * (self.x - t)) ** 4
        self.init = self.analytical(0.0)
        self.exact = self.analytical(self.t_max)
        # self.limit_fnc = self.superbee
        # self.flux_fnc = self.flux_limited
        # self.time_int = RK()
        self.addme = []

    # def superbee(self, t, r):
    #     return np.maximum(np.maximum(np.zeros_like(r), np.minimum(2.0 * r, 1.0)), np.minimum(r, 2.0))

    # def centered_sch(self, delta_t, t, now, dir):
    #     return dir * np.roll(now, 0)

    # def upwind_sch(self, delta_t, t, now, dir):
    #     return now

    # def lax_wendroff_sch(self, delta_t, t, now, dir):
    #     flux = self.centered_sch(delta_t, t, now, dir) - self.v * \
    #         ((delta_t) / (2 * self.delta_x)) * \
    #         (1 * dir * now)
    #     return flux

    # def flux_limited(self, delta_t, t, now, dir):
    #     mesh_grad = now - np.roll(now, 1)
    #     r = np.nan_to_num(mesh_grad / np.roll(mesh_grad, -dir))
    #     limiter = self.limit_fnc(t, r)
    #     _DEBUG()
    #     upwind = self.upwind_sch(delta_t, t, now, dir)
    #     lax = self.lax_wendroff_sch(delta_t, t, now, dir)
    #     return upwind + np.roll(limiter, -dir) * (lax - upwind)

    def boundary_cond(self, t, ghosts):
        ghosts[0] = ghosts[-3]
        ghosts[1] = ghosts[-4]  # np.sin(30 * t)
        ghosts[-2] = ghosts[3]
        ghosts[-1] = ghosts[2]
        return ghosts

    def split_velocity(self, v):
        m = np.max(np.abs(v))
        # leftwards = 0.5 * (v + m)
        # rightwards = np.roll(-0.5 * (v - m), 1)
        leftwards = np.where(v > 0, v, 0)
        rightwards = np.roll(-np.where(v < 0, v, 0), 1)
        return leftwards, rightwards

    @autojit()
    def spatial_deriv(self, delta_t, t, now):
        now = self.boundary_cond(t, now)
        leftwards_v, rightwards_v = self.split_velocity(self.v)
        left_flux = leftwards_v * self.flux_fnc(delta_t, t, now, -1)
        right_flux = rightwards_v * self.flux_fnc(delta_t, t, now, 1)
        total_left_flux = -(left_flux - np.roll(right_flux, 1))
        derivative = (total_left_flux - np.roll(total_left_flux, -1)) / self.delta_x
        return derivative

    def _compute(self, ax, sym):
        ax.plot(self.x, self.init)
        p1, = ax.plot(self.x, self.init, sym)
        error_fig = pyp.figure()
        error_fig = error_fig.add_subplot(111)
        pyp.show(block=False)

        result = np.pad(self.init.copy(), 2, 'constant')
        self.t = self.delta_t
        # self.time_int.setup(self.spatial_deriv)
        diff_old = result
        diff = result
        p_error, = error_fig.plot(diff_old)
        while self.t <= self.t_max:
            f = lambda adt, x: self.spatial_deriv(adt * self.delta_t, self.t, x)
            dt = self.delta_t
            t1 = result + 0.391752226571890 * dt * f(0.391752226571890, result)
            t2 = 0.444370493651235 * result + \
                0.555629506348765 * t1 + \
                0.368410593050371 * dt * f(0.368410593050371, t1)
            t3 = 0.620101851488403 * result + \
                0.379898148511597 * t2 + \
                0.251891774271694 * dt * f(0.251891774271694, t2)
            t4 = 0.178079954393132 * result + \
                0.821920045606868 * t3 + \
                0.544974750228521 * dt * f(0.544974750228521, t3)
            result = 0.517231671970585 * t2 + \
                0.096059710526147 * t3 + \
                0.063692468666290 * dt * f(0.063692468666290, t3) + \
                0.386708617503269 * t4 + \
                0.226007483236906 * dt * f(0.226007483236906, t4)
            # temp1 = result + self.delta_t * self.spatial_deriv(self.delta_t, self.t, result)
            # temp2 = 0.75 * result + \
            #     0.25 * temp1 + \
            #     0.25 * self.delta_t * self.spatial_deriv(0.25 * self.delta_t, self.t, temp1)
            # result = (1.0 / 3.0) * result + \
            #           (2.0 / 3.0) * temp2 + \
            #           (2.0 / 3.0) * self.delta_t * self.spatial_deriv((2.0 / 3.0) * self.delta_t, self.t, temp2)
            # result = result + dt * f(result)
            do_we_plot = self.t / self.plot_interval
            self.t += self.delta_t
            exact = self.analytical(self.t)
            diff = result[2:-2] - exact
            error = np.sum(diff ** 2) / len(exact)
            time.sleep(0.1)
            # if error >= 0.035241257075:
            # # if error >= 0.00035241257075:
            #     p1.set_ydata(result)
            #     pyp.draw()
            #     pyp.figure()
            #     pyp.plot(self.x, diff)
            #     pyp.plot(self.x, diff_old)
            #     pyp.show()
            #     _DEBUG(5)
            p_error.set_ydata(diff)
            pyp.draw()
            diff_old = diff
            print error
            if (not self.always_plot) and \
                    abs(do_we_plot - round(do_we_plot)) > (self.delta_t / self.plot_interval):
                continue
            if self.plot:
                print self.t
                p1.set_ydata(result[2:-2])
                pyp.draw()
        ax.plot(self.x, result, 'o')
        ax.plot(self.x, self.exact)
        # return deriv

# def test_varying_grid():
#     test_case = 'jumpy'
#     grid1 = np.linspace(-5, 5, 100)
#     fvm1 = FVM(test_case, grid1)
#     delta_x = np.ones(100) * 0.1
#     grid2 = np.cumsum(delta_x)
#     fvm2 = FVM(test_case, grid2)


if __name__ == "__main__":
    setup_debug()
    fig = pyp.figure()
    fvm = FVM('jumpy')
    ax = fig.add_subplot(111)
    ax.axis((0, fvm.x_max, -1.5, 1.5))
    fvm.flux_fnc = WENO().compute
    retval1 = fvm._compute(ax, '.')
    pyp.show()






    # # fig2 = pyp.figure()
    # fvm.flux_fnc = fvm.flux_limited
    # retval2 = fvm._compute(ax, '+')
    # # fvm.flux_fnc = fvm.upwind_sch
    # # retval2 = fvm._compute(ax, 'x')
    # # fvm.flux_fnc = fvm.centered_sch
    # # retval2 = fvm._compute(ax, '+')
    # fvm.flux_fnc = fvm.lax_wendroff_sch
    # retval3 = fvm._compute(ax, 'o')
    # # pyp.show()
    # # pyp.plot(retval1 - retval3)
