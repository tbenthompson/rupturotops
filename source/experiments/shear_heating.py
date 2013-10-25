import numpy as np
from matplotlib import pyplot as pyp
import scipy.integrate
from core.debug import _DEBUG
from core.experiment import Experiment
from core.constants import consts


class ShearHeating(Experiment):

    def _initialize(self):
        """
        Nondimensionalizes all the inputs. All the relevant scales are stored.
        """
        self.data.length_scale = np.sqrt(self.material.density /
                                         self.material.shear_modulus) * \
            self.material.thermal_diffusivity
        self.data.time_scale = self.material.density * self.material.thermal_diffusivity / \
            self.material.shear_modulus
        self.data.stress_scale = self.material.shear_modulus
        self.data.inv_prandtl = self.material.creep_constant * self.material.density * \
            self.material.thermal_diffusivity * \
            self.material.shear_modulus ** (self.material.stress_exponent - 1)
        self.data.eckert = self.material.shear_modulus / (self.params.delta_temp *
                                                          self.material.specific_heat *
                                                          self.material.density)

        self.data.x = self.params.x / self.data.length_scale
        self.data.delta_x = abs(self.data.x[1] - self.data.x[0])
        self.data.t = self.params.t / self.data.time_scale
        self.data.initial_temp = (self.params.initial_temp - self.params.low_temp) / self.params.delta_temp
        self.data.stress = self.params.stress / self.data.stress_scale
        self.data.source_term = self.params.source_term * self.data.time_scale / self.params.delta_temp

    def _compute(self):
        self.calc_temp()
        self.calc_strain()

    def _arrhenius_power(self, T):
        T_expr = self.params.low_temp + self.params.delta_temp * T
        coefficient = self.material.activation_energy / consts.R
        return -coefficient / T_expr

    def calc_temp(self):
        """
        Runs the centered space finite difference method. This is the core method.

        include_shear_heating=True implies that the shear heating term should be included.

        Dirichlet boundary conditions are applied at the boundaries. The value remains
        equal to whatever it was in the initial conditions

        ADD EQUATION THAT IT IS SOLVING in MathJax
        """
        # _DEBUG()
        def rhs(current_temp, t):
            print("Current time step: " + str(t))
            print("Final time: " + str(self.data.t[-1]))

            exp_term = self.data.inv_prandtl * self.data.eckert * \
                self.data.stress ** (self.material.stress_exponent + 1) * \
                np.exp(self._arrhenius_power(current_temp[1:-1]))
            _DEBUG()

            diffusion_term = (current_temp[:-2] - 2 * current_temp[1:-1]
                              + current_temp[2:]) / (self.data.delta_x ** 2)

            retval = np.zeros(self.data.x.shape)
            retval[1:-1] += diffusion_term + exp_term + self.data.source_term[1:-1]
            return retval

        #The odeint function utilizes LSODA in ODEPACK, a fortran routine that alternates between
        #various multistep methods depending on the estimated stiffness of the equation.
        self.data.temp_time = scipy.integrate.odeint(rhs, self.data.initial_temp, self.data.t)

    def calc_strain(self):
        """
        Integrates the temperature-time data to find the strain-time data.
        Uses simpson's rule.
        ADD EQUATION THAT IT SOLVES
        """
        tointegrate = self.data.inv_prandtl * \
            self.data.stress ** self.material.stress_exponent * \
            np.exp(self._arrhenius_power(self.data.temp_time))
        self.data.strain_time = scipy.integrate.simps(tointegrate.T, self.data.t)

    def _visualize(self):
        if self.params.plot_init_cond:
            self.plot_init_cond()
        if self.params.plot_temp_time:
            self.plot_temp_time()
        if self.params.plot_strain_time:
            self.plot_strain_time()

    def plot_init_cond(self):
        pyp.plot(self.data.x, self.data.initial_temp)
        pyp.show()

    def plot_temp_time(self):
        selection = range(0, len(self.data.t), self.params.plot_every)
        pyp.plot(self.data.x, self.data.temp_time[selection, :].T, '-')
        pyp.show()

    def plot_strain_time(self):
        pass
