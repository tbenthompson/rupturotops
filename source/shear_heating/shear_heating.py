import numpy as np
from matplotlib import pyplot as pyp
import scipy.integrate
# from debug import _DEBUG
from shear_heating_params import params
from core.experiment import Experiment

class ShearHeating(Experiment):

    def __init__(self):
        params.t = np.linspace(0,
                                  params.t_max,
                                  params.steps)

        half_width = params.points * params.delta_x / 2.0
        params.X = np.linspace(-half_width,
                                  half_width,
                                  params.points)
        params.temp_scale, \
            params.time_scale, \
            params.the_constant = find_consts(params.stress,
                                                 params.length_scale,
                                                 params.material)
        params.min_temp = params.min_temp / params.temp_scale  # Kelvins
        params.temp_mass = params.temp_mass / params.temp_scale
        params.initial_temp = self.gaussian_temp(params)


    # boundary conditions are dirichlet and specified as the
    # first and last component of initial_temp
    def gaussian_initial_conditions(self):
        temp = (params.temp_mass / np.sqrt(np.pi)) * \
            np.exp(-params.gaussian_width * params.X ** 2)
        temp += params.min_temp
        return temp.T


    def fdm_run(data):
        """
        Runs the centered space finite difference method. This is the core method.

        include_shear_heating=True implies that the shear heating term should be included.

        Dirichlet boundary conditions are applied at the boundaries. The value remains
        equal to whatever it was in the initial conditions
        """
        # _DEBUG()
        def rhs(current_temp, t):
            # print t
            # if data.plot_every is not None and \
                # step % data.plot_every == 0:
            # pyp.plot(data.X, current_temp)

            exp_term = 0
            if data.include_shear_heating:
                exp_term = data.the_constant * \
                    np.exp(-1 / (current_temp[1:-1]))

            diffusion_term = (current_temp[:-2] - 2 * current_temp[1:-1]
                              + current_temp[2:]) / (data.delta_x ** 2)

            retval = np.zeros((len(current_temp)))
            retval[1:-1] = diffusion_term + exp_term
            return retval

        return scipy.integrate.odeint(rhs, data.initial_temp, data.t)



    def finish_calc_params(params):
        # assign_data_location(params)


    def find_consts(stress, length_scale, rock_params):
        temp_scale = rock_params.activation_energy / rock_params.R
        time_scale = length_scale ** 2 / rock_params.thermal_diffusivity
        numer = (time_scale * rock_params.creep_constant
                 * stress ** (rock_params.stress_exponent + 1))
        denom = (temp_scale * rock_params.specific_heat * rock_params.density)
        return temp_scale, time_scale, numer / denom

    def calc_strain(temp, data):
        tointegrate = data.material.creep_constant * data.time_scale * \
            data.stress ** data.material.stress_exponent * \
            np.exp(-1 / temp)
        strain = scipy.integrate.cumtrapz(tointegrate,
                                          dx=data.t_max / data.steps)
        return strain

    def total_strain_plot():
        finish_calc_params(params)
        data = params.copy()
        pyp.plot(data.X, data.initial_temp, '.')
        pyp.show()
        # exit()
        final_temp_exp = run(data)
        exp_strain = calc_strain(final_temp_exp.T, data)[:, -1]

        # pyp.plot(data.X, final_temp_exp.T)
        # pyp.show()

        data.include_shear_heating = False
        final_temp_no_exp = run(data)
        no_exp_strain = calc_strain(final_temp_no_exp.T, data)[:, -1]

        strain_diff = exp_strain - no_exp_strain
        pyp.plot(data.X, strain_diff)
        pyp.plot(data.X, exp_strain)
        pyp.show()
        print np.max(strain_diff)
        data.results = dict()
        data.results.final_temp_exp = final_temp_exp
        data.results.final_temp_no_exp = final_temp_no_exp
        data.results.exp_strain = exp_strain
        data.results.no_exp_strain = no_exp_strain
        data.results.strain_diff = strain_diff


if __name__ == "__main__":
    # compare_exp_and_no_exp()
    total_strain_plot()



def test_gaussian_temp():
    params = dict()
    params.X = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    params.temp_mass = 1
    params.min_temp = 0
    params.gaussian_width = 1
    temp = gaussian_temp(params)
    assert temp[1] < temp[2]
    assert temp[2] > temp[3]
    assert temp[3] > temp[4]
    #distance = 0.5 , distance ** 2 = 0.25
    correct = (0.7788007830714049 / 1.7724538509055159)
    assert (temp[0] - correct) < 0.000001

    params.temp_mass = sqrt(pi)
    temp2 = gaussian_temp(params)
    assert temp2[2] == 1

    params.gaussian_width = 2
    temp3 = gaussian_temp(params)
    assert temp3[0] == exp(-0.5)

    params.min_temp = 1
    params.temp_mass = 2 * sqrt(pi)
    temp4 = gaussian_temp(params)
    assert temp4[2] == 3



def test_calc_stran():
    temp = np.array([-1.0, -1.0, -1.0])
    data = dict()
    data.stress = 1.0
    data.material = dict(creep_constant=1.0, stress_exponent=1.0)
    data.time_scale = 1.0
    data.steps = 2.0
    data.t_max = 2.0
    strain = calc_strain(temp.T, data)
    assert (strain == [exp(1), exp(2)]).all



