from core.data_controller import DataController

#ALL IN STANDARD SI UNITS
# all values here come from the wet diabase in takeuchi, fialko
wetdiabase = DataController()
wetdiabase.density = 2850  # kg/m^3
wetdiabase.specific_heat = 1000  # J/kgK
wetdiabase.activation_energy = 2.6e5  # J/mol
wetdiabase.stress_exponent = 3.4
wetdiabase.creep_constant = 2.2e-4 * 10 ** (-6 * 3.4)  # (Pa^-n)/sec
wetdiabase.thermal_diffusivity = 7.37e-7  # m^2/sec
wetdiabase.youngs_modulus = 80e9  # Pa
wetdiabase.poisson = 0.25
wetdiabase.shear_modulus = wetdiabase.youngs_modulus / (2 * (1 + wetdiabase.poisson))
