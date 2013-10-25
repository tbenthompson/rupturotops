import numpy as np
from matplotlib import pyplot as pyp
import pylab
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from core.debug import _DEBUG
from scipy.integrate import quad
from core.constants import consts
from core.experiment import Experiment


class Diffusive(Experiment):
    """
    Computes the analytical gaussian result for a dirac delta heat source at the fault
    and then computes the strain to be expected if we ignore stress and shear heating
    effects. This should only be used as a metric against which to make sure other
    experiments including more of the relevant factors are behaving correctly

    Add the equaton I'm solving in MathJax.
    """
    def _initialize(self):
        self.data.inv_prandtl = self.material.creep_constant *\
            self.material.density *\
            self.material.thermal_diffusivity *\
            self.material.shear_modulus ** self.material.stress_exponent
        self.data.time_scale = self.params.time_scale
        self.data.length_scale = np.sqrt(self.material.thermal_diffusivity * self.data.time_scale)

        self.data.x_domain = self.params.x_domain / self.data.length_scale
        self.data.t_domain = self.params.t_domain / self.data.time_scale
        print('Time Scale: ' + str(self.data.time_scale))
        print('Length Scale: ' + str(self.data.length_scale))
        print('1/Pr: ' + str(self.data.inv_prandtl))
        pass

    def _compute(self):
        # self.calc_strain()
        pass

    def temp_fnc(self, x, t):
        def gaussian_temp(x, t):
            return np.exp(-(x ** 2) / (4 * t)) / \
                np.sqrt(4 * np.pi * t)
        returnvalue = 0
        cycle = np.int(np.floor(t))
        for i in range(0, 1 + cycle):
            returnvalue += gaussian_temp(x, t - i)
        return self.params.delta_t * returnvalue + self.params.init_temp

    def strainrate_fnc(self, x, t):
        return self.data.inv_prandtl * np.exp(-(self.material.activation_energy / consts.R) /
                                              self.temp_fnc(x, t))

    def calc_strain(self):
        """
        Simply integrates the analytically derived temperature profile using
        a arrhenius form strain rate equation to get the strain.
        """
        self.data.strain = np.zeros((len(self.data.x_domain), len(self.data.t_domain)))
        for i in range(0, len(self.data.x_domain)):
            print("Calculating: x = " + str(self.data.x_domain[i]))
            for j in range(1, len(self.data.t_domain)):
                self.data.strain[i, j] = self.data.strain[i, j - 1] + \
                    quad(lambda t: self.strainrate_fnc(self.data.x_domain[i], t),
                         self.data.t_domain[j - 1], self.data.t_domain[j])[0]

    def _visualize(self):
        # pyp.plot(self.data.x_domain, map(lambda x: self.temp_fnc(x, 10000), self.data.x_domain))
        # pyp.show()
        self.plot_temp_strainrate()
        # self.plot_strain_map()
        pass

    def plot_temp_strainrate(self):
        host = host_subplot(111, axes_class=AA.Axes)
        pyp.subplots_adjust(right=0.75)
        par1 = host.twinx()
        host.set_xlabel(r'$\frac{t}{P}$')
        host.set_ylabel(r'$\frac{T - T_i}{\Delta T}$')
        par1.set_ylabel(r'$\epsilon$')
        p1, = host.plot(self.data.t_domain, map(lambda t: self.temp_fnc(0.5, t), self.data.t_domain + 1),
                  label="Temp")
        p2, = par1.plot(self.data.t_domain, map(lambda t: self.strainrate_fnc(0.5, t), self.data.t_domain + 1),
                  label="Strainrate")
        par1.set_ylim(auto=True)
        host.legend()
        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
        pyp.draw()
        pyp.show()

    def plot_strain_map(self):
        """
        Plots the log of the strain as a function of x and t using a colormap.
        """
        log_strain = np.log(self.data.strain[:, 1:])
        bounds = (self.data.t_domain[0], self.data.t_domain[-1],
                  self.data.x_domain[0], self.data.x_domain[-1])
        colmap = pylab.get_cmap('winter')
        pyp.imshow(np.flipud(log_strain), cmap=colmap,
                   aspect='auto', extent=bounds,
                   interpolation='nearest')
        pyp.colorbar(ticks = np.linspace(np.min(np.min(log_strain)), np.max(np.max(log_strain)), 10))
        pyp.xlabel('t/P')
        pyp.ylabel('x/L')
        pyp.title('Strain')
        pyp.show()
