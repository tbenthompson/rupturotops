from data_controller import DataController

#ALL IN STANDARD SI UNITS
# all values here come from the wet diabase in takeuchi, fialko
wetdiabase = DataController()
wetdiabase.density = 2850  # kg/m^3
wetdiabase.specificheat = 1000  # J/kgK
wetdiabase.activationenergy = 2.6e5  # J/mol
wetdiabase.stressexponent = 3.4
wetdiabase.creepconstant = 2.2e-4 * 10 ** (-6 * 3.4)  # (Pa^-n)/sec
wetdiabase.thermaldiffusivity = 7.37e-7  # m^2/sec
wetdiabase.youngsmodulus = 80e9  # Pa
wetdiabase.poisson = 0.25
