import numpy as np
from matplotlib import pyplot as pyp
from core.debug import _DEBUG

def view_soln(Szx, Szy):
    pyp.figure(1)
    pyp.imshow(np.log(np.abs(Szx)))
    pyp.colorbar()
    pyp.figure(2)
    pyp.imshow(np.log(np.abs(Szy)))
    pyp.colorbar()
    pyp.show()

def elastic_stress_soln(x, y, s, D, shear_modulus):
    """
    Use the elastic half-space stress solution from Segall (2010)
    """
    # TEST THIS!
    factor = (s * shear_modulus) / (2 * np.pi)
    main_term = -(y - D) / ((y - D) ** 2 + x ** 2)
    image_term = (y + D) / ((y + D) ** 2 + x ** 2)
    Szx = factor * (main_term + image_term)

    main_term = x / (x ** 2 + (y - D) ** 2)
    image_term = -x / (x ** 2 + (y + D) ** 2)
    Szy = factor * (main_term + image_term)
    return Szx, Szy

def eff_stress(Szx, Szy):
    return np.sqrt(Szx ** 2 + Szy ** 2)

class StressSolver(object):
    """
    Controls solution of the stress ODE and initial conditions
    """

    def __init__(self, params):
        self.params = params
        self.material = params.material

        self.X_ = np.linspace(
            self.params.x_min, self.params.x_max, self.params.x_points)
        self.Y_ = np.linspace(
            self.params.y_min, self.params.y_max, self.params.y_points)
        self.X, self.Y = np.meshgrid(self.X_, self.Y_)
        self.Szx, self.Szy = elastic_stress_soln(
            self.X, self.Y, self.params.fault_slip, self.params.fault_depth,
            self.material.shear_modulus)
        self.initialSzx = self.Szx.copy()
        self.initialSzy = self.Szy.copy()
        # pyp.imshow(np.log(self.stress))
        # pyp.show()

    def inv_eff_visc(self, Szx, Szy, temp):
        retval = np.where(self.Y > self.params.elastic_depth,
                          1.0 / self.params.viscosity, 0.0)
        # retval = self.material.creep_constant * \
        #     eff_stress(Szx, Szy) ** (self.material.stress_exponent - 1) * \
        #     dfn.exp(-(self.material.activation_energy / consts.R) / temp)
        return retval

    def deriv_momentum(self, t, Szx, Szy):
        factor = -(self.material.shear_modulus *
                   self.inv_eff_visc(Szx, Szy, self.params.background_temp))
        dSzx = factor * Szx
        dSzy = factor * Szy
        return dSzx, dSzy

    def update_momentum(self, t, dt):
        pyp.figure(1)
        pyp.plot(self.X_, self.Szx[25, :])
        pyp.figure(2)
        pyp.plot(self.X_, self.Szy[25, :])
        # self.stress_solver.set_initial_value(self.stress, t)
        sol = []
        # warnings.filterwarnings("ignore", category=UserWarning)
        internal_t = t
        internal_dt = dt / 100.0
        while internal_t < t + dt:
            dSzx, dSzy = self.deriv_momentum(
                internal_t, self.Szx, self.Szy)
            self.Szx += internal_dt * dSzx
            self.Szy += internal_dt * dSzy
            internal_t += internal_dt
            # sol.append([self.stress_solver.t, self.stress_solver.y])
            # print("Time: " + str(internal_t))
            # print("Stress: " + str(deltaSzx))
        # self.stress = sol[-1][1]
        # warnings.resetwarnings()
        # view_soln(self.Szx, self.Szy)
        return self.Szx, self.Szy

    def helmholtz_projection(self, t, dt, dvdx, dvdy):
        self.Szx += self.material.shear_modulus * dt * dvdx
        self.Szy += self.material.shear_modulus * dt * dvdy

