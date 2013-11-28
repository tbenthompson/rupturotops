import numpy as np
from matplotlib import pyplot as pyp
import scipy.integrate
from core.debug import _DEBUG
from core.experiment import Experiment
from core.constants import consts


class ShearHeating(Experiment):
    def _initialize(self):
        pass

    def _compute(self):
        self.calc_temp()
        self.calc_strain()

    def _arrhenius_power(self, T):
        T_expr = self.params.low_temp + self.params.delta_temp * T
        coefficient = self.material.activation_energy / consts.R
        return -coefficient / T_expr


    def eff_visc(self, stress, temp):
        return self.material.creep_constant * \
            stress ** (self.material.stress_exponent - 1) * \
            np.exp(-(self.material.activation_energy / consts.R) / temp)

    def time_step(current_temp, t):
        print("Current time step: " + str(t))
        print("Final time: " + str(self.data.t[-1]))
        diffusion_term = (current_temp[:-2] - 2 * current_temp[1:-1]
                          + current_temp[2:]) / (self.data.delta_x ** 2)
        shear_heat_term =

        retval = np.zeros(self.data.x.shape)
        retval[1:-1] += diffusion_term + \
            exp_term + self.data.source_term[1:-1]
        return retval

    def calc_temp(self):
        """
        Uses LSODA from ODEPACK to perform adaptive time stepping.
        """
        self.data.temp_time = \
            scipy.integrate.odeint(time_step,
                                   self.data.initial_temp,
                                   self.data.t)

    def calc_strain(self):
        """
        Integrates the temperature-time data to find the strain-time data.
        Uses simpson's rule.
        ADD EQUATION THAT IT SOLVES
        """
        tointegrate = self.data.inv_prandtl * \
            self.data.stress ** self.material.stress_exponent * \
            np.exp(self._arrhenius_power(self.data.temp_time))
        self.data.strain_time = scipy.integrate.simps(
            tointegrate.T, self.data.t)

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
