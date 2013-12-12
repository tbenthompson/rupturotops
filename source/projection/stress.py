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

        #create a staggered grid for Szx
        self.SzxX_ = np.linspace(self.params.x_min,
                                 self.params.x_max - \
                                 self.params.delta_x,
                                 self.params.x_points - 1)
        self.SzxY_ = np.linspace(self.params.y_min,
                                 self.params.y_max,
                                 self.params.y_points)
        self.SzxX_ += (self.params.delta_x / 2.0)
        self.SzxX, self.SzxY = np.meshgrid(self.SzxX_, self.SzxY_)

        #create a staggered grid for Szy
        self.SzyX_ = np.linspace(self.params.x_min,
                                 self.params.x_max,
                                 self.params.x_points)
        self.SzyY_ = np.linspace(self.params.y_min,
                                 self.params.y_max - \
                                 self.params.delta_y,
                                 self.params.y_points - 1)
        self.SzyY_ += (self.params.delta_y / 2.0)
        self.SzyX, self.SzyY = np.meshgrid(self.SzyX_, self.SzyY_)
#
#         def slip_func(y):
#             return np.where(y < self.params.fault_depth,
#                 np.cos((y * np.pi) / (self.params.fault_depth * 2)), 0.0)
#         self.Szx, throwaway = elastic_half_space_stress(slip_func,
#                                            self.params.fault_depth,
#                                            self.SzxX, self.SzxY,
#                                            self.material.shear_modulus)
#         throwaway, self.Szy = elastic_half_space_stress(slip_func,
#                                            self.params.fault_depth,
#                                            self.SzyX, self.SzyY,
#                                            self.material.shear_modulus)

        self.Szx, throwaway = elastic_stress(
            self.SzxX, self.SzxY, self.params.fault_slip, self.params.fault_depth,
            self.material.shear_modulus)

        throwaway, self.Szy = elastic_stress(
            self.SzyX, self.SzyY, self.params.fault_slip, self.params.fault_depth,
            self.material.shear_modulus)
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

    def inv_eff_visc(self, X, Y, Szx, Szy, temp):
        """
        Computes the inverse effective viscosity. This should be updated
        to work with the staggered grid approach.
        """
        retval = np.where(Y > self.params.elastic_depth,
                          1.0 / self.params.viscosity, 0.0)
        # retval = self.material.creep_constant * \
        #     eff_stress(Szx, Szy) ** (self.material.stress_exponent - 1) * \
        #     dfn.exp(-(self.material.activation_energy / consts.R) / temp)
        return retval

    def deriv_momentum(self, t, Szx, Szy):
        """
        Momentum derivative for the ODE part of the projection method. Should
        be updated to work with the staggered grid approach when viscosity is
        stress dependent. Also, be careful when coupling to temperature because of
        staggered grid.
        """
        factorSzx = -(self.material.shear_modulus *
                   self.inv_eff_visc(self.SzxX, self.SzxY, Szx, Szy, self.params.background_temp))
        factorSzy = -(self.material.shear_modulus *
                   self.inv_eff_visc(self.SzyX, self.SzyY, Szx, Szy, self.params.background_temp))
        dSzx = factorSzx * Szx
        dSzy = factorSzy * Szy
        return dSzx, dSzy

    def update_momentum(self, t, dt):
        pyp.figure(1)
        pyp.plot(self.SzxX_, self.Szx[0, :])
        pyp.figure(2)
        pyp.plot(self.SzyX_, self.Szy[0, :])
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

    def helmholtz_projection(self, t, dt, v):
        dvdx = (v[:, 1:] - v[:, :-1]) / self.params.delta_x
        dvdy = (v[1:, :] - v[:-1, :]) / self.params.delta_y
        self.Szx += self.material.shear_modulus * dt * dvdx
        self.Szy += self.material.shear_modulus * dt * dvdy

