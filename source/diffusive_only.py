import numpy as np
from matplotlib import pyplot as pyp
import pylab
# from pdb import set_trace as _DEBUG
from scipy.integrate import quad
from parameters import params
import multiprocessing

multiprocessing.Pool(4)

exit(1)

low_x = 0.0
high_x = 1.0
low_t = 0.01
high_t = 1.0
x_count = 500
t_count = 500
x_domain = np.linspace(low_x, high_x, x_count)
t_domain = np.linspace(low_t, high_t, t_count)
#assume N = 1

P = 100 * 365 * 24 * 3600
N = P * params['material']['creepconstant'] * params['stress'] ** params['material']['stressexponent']
lambduh = params['material']['activationenergy'] / params['material']['R']
l = np.sqrt(params['material']['thermaldiffusivity'] * P)

T_i = 673 / lambduh
T_f = 676 / lambduh
load = False
strain_load_filename = './strains.npy'
strain_save_filename = './strains'


print('P: ' + str(P))
print('l: ' + str(l))
print('N: ' + str(N))



def temp_fnc(x, t):
    return ((T_f - T_i) * np.exp(-(x ** 2) / (4 * t)) /
                  np.sqrt(4 * np.pi * t))


def strainrate_fnc(x, t):
    return N * np.exp(-1 / (T_i + temp_fnc(x, t)))


def calculate_strain():
    strain = np.zeros((len(x_domain), len(t_domain)))
    for i in range(0, len(x_domain)):
        for j in range(1, len(t_domain)):
            strain[i, j] = strain[i, j - 1] + \
                quad(lambda t: strainrate_fnc(x_domain[i], t),
                     t_domain[j - 1], t_domain[j])[0]
    np.save(strain_save_filename, strain)
    return strain

if load:
    strain = np.load(strain_load_filename)
else:
    pass

total_temp = 0
for i in range(0,10000):
    if i % 1000 == 0:
        print total_temp
    total_temp += temp_fnc(0.1, float(i) + 0.5)
# total_temp /= params['material']['R']/params['material']['activationenergy']
    # strain = calculate_strain()
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
