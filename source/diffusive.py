import numpy as np
from matplotlib import pyplot as pyp
import pylab
# from pdb import set_trace as _DEBUG
from scipy.integrate import quad


class Diffusive(Experiment):
    """
    Computes the analytical gaussian result for a dirac delta heat source at the fault
    and then computes the strain to be expected if we ignore stress and shear heating
    effects. This should only be used as a metric against which to make sure other
    experiments including more of the relevant factors are behaving correctly
    """
    def _initialize(self):
        N = P * params.material.creep_constant * params.stress ** params.material.stress_exponent
        lambduh = params.material.activation_energy / params.material.R
        l = np.sqrt(params.material.thermal_diffusivity * P)
        T_i = 673 / lambduh
        T_f = 676 / lambduh
        print('P: ' + str(P))
        print('l: ' + str(l))
        print('N: ' + str(N))

        pass

    def _compute(self):
        def temp_fnc(x, t):
            return ((T_f - T_i) * np.exp(-(x ** 2) / (4 * t)) /
                          np.sqrt(4 * np.pi * t))

        def strainrate_fnc(x, t):
            return N * np.exp(-1 / (T_i + temp_fnc(x, t)))


        strain = np.zeros((len(self.data.x_domain), len(self.data.t_domain)))
        for i in range(0, len(self.data.x_domain)):
            for j in range(1, len(self.data.t_domain)):
                strain[i, j] = strain[i, j - 1] + \
                    quad(lambda t: strainrate_fnc(self.data.x_domain[i], t),
                         self.data.t_domain[j - 1], self.data.t_domain[j])[0]
        np.save(strain_save_filename, strain)

        # total_temp = 0
        # for i in range(0,10000):
        #     if i % 1000 == 0:
        #         print total_temp
        #     total_temp += temp_fnc(0.1, float(i) + 0.5)
        # total_temp /= params.material.R/params.material.activationenergy
            # strain = calculate_strain()
        return strain

    def _visualize(self):
        # pyp.plot(t_domain, temp_fnc(0, t_domain))
        # pyp.plot(t_domain, strainrate_fnc(0.1, t_domain))
        # pyp.plot(t_domain, strain[:, 5])
        # pyp.show()

        # ignore first index because log(0) is bad
        # log_strain = np.log(strain[:50, 1:50])
        # bounds = (low_t, high_t, low_x, high_x)
        # colmap = pylab.get_cmap('winter')
        # pyp.imshow(log_strain, cmap=colmap,
        #            aspect='auto', extent=bounds,
        #            interpolation='nearest')
        # pyp.colorbar(ticks = np.linspace(np.min(np.min(log_strain)), np.max(np.max(log_strain)), 10))
        # pyp.show()


        #assume N = 1
        pass
