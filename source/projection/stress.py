import numpy as np
from matplotlib import pyplot as pyp
from core.debug import _DEBUG
from linear_viscoelastic.simple_elastic import elastic_stress
from linear_viscoelastic.full_elastic import elastic_half_space_stress

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
        def slip_func(y):
            return np.where(y < self.params.fault_depth,
                np.cos((y * np.pi) / (self.params.fault_depth * 2)), 0.0)
        self.Szx, self.Szy = elastic_half_space_stress(slip_func,
                                           self.params.fault_depth,
                                           self.X, self.Y,
                                           self.material.shear_modulus,
                                                       )

        # self.Szx, self.Szy = elastic_stress(
        #     self.X, self.Y, self.params.fault_slip, self.params.fault_depth,
        #     self.material.shear_modulus)

        self.initialSzx = self.Szx.copy()
        self.initialSzy = self.Szy.copy()

    @staticmethod
    def view_soln(Szx, Szy):
        pyp.figure(1)
        pyp.imshow(np.log(np.abs(Szx)))
        pyp.colorbar()
        pyp.figure(2)
        pyp.imshow(np.log(np.abs(Szy)))
        pyp.colorbar()
        pyp.show()

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
        # pyp.figure(1)
        # pyp.plot(self.X_, self.Szx[25, :])
        # pyp.figure(2)
        # pyp.plot(self.X_, self.Szy[25, :])
        # self.stress_solver.set_initial_value(self.stress, t)
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

