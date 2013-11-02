import numpy as np
import matplotlib.pyplot as pyp
from parameters.material import wetdiabase
assert(wetdiabase)
from math import exp
assert(exp)
from core.constants import consts
assert(consts)
from scipy.special import erf

#################################
# Compute the the two ratios in the away-from-fault non-dimensional
# viscoelastic wave equations.
#################################
# delta_T = 500  # kelvins
# deviatoric_stress = 100e6
# geotherm_temperature = 600  # kelvin
# temp = 0.5
#
# temp_exp = exp(-(wetdiabase.activation_energy / consts.R) * (1 / (geotherm_temperature + delta_T * temp)))
#
# one_over_prandtl = wetdiabase.density * wetdiabase.thermal_diffusivity * \
#     wetdiabase.creep_constant * wetdiabase.shear_modulus ** (wetdiabase.stress_exponent - 1)
# eckert = wetdiabase.shear_modulus / (delta_T * wetdiabase.specific_heat * wetdiabase.density)
# prandtl = 1 / one_over_prandtl
#
# #incorporate the fact that I scaled so that stress is small, not between 0 and 1.
# #also incorporate the effects of the arrhenius term
# true_prandtl = prandtl * (1 / temp_exp)
# viscosity = wetdiabase.creep_constant * wetdiabase.shear_modulus ** (wetdiabase.stress_exponent - 1) * \
#     (1 / temp_exp)
#
# print wetdiabase.density * wetdiabase.thermal_diffusivity / deviatoric_stress
# print viscosity
# print one_over_prandtl
# print prandtl
# print true_prandtl
# print eckert
#


################################
# Compute the average number of cells
# needed in a 1D simulation where cells are
# apportioned as a function of the temperature
################################

def abc():
    x = np.linspace(-1000, 0, 10)
    y = np.exp(-x**2 / 1000000)
    factor = np.sqrt(1000000)
    y = 1 + 1000 ** (1 + erf(x / factor)) / 10
    print sum(y)
    pyp.plot(x, y)
    pyp.show()
