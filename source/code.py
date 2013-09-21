from math import pi,factorial   
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as pyp




# displacement = []
# displacement2 = []
# for i in range(len(x)):
#     displacement.append(elastic_half_space(lambda z: 1.0, x[i]))
#     displacement2.append(two_layer_elastic(lambda z: 1.0, 1.0, 0.0, x[i]))
# pyp.plot(x,displacement)
# pyp.show()
# pyp.plot(x, displacement2)
# pyp.show()

alpha = 1.0
a = -1.0

import constantslip
displacement2, x2 = constantslip.standard_parameters_test(False)
# x = np.arange(-(100.0/15.0), (100.0/15.0), (200/(15.0*201)))
x = x2/15.0
t = np.arange(0,5)

displacement = []
for i in range(len(t)):
    displacement.append([])
    for j in range(len(x)):
        displacement[i].append(constant_slip_constant_shear_modulus_viscoelastic(x[j], t[i], alpha, a))
pyp.plot(x, displacement[0], x, displacement[1], x, displacement[2], x, displacement[3])
pyp.axis((-(100/15.0),(100/15.0), -1, 1))
pyp.xlabel("distance from the fault")
pyp.ylabel("displacement")
pyp.show()

print len(displacement2)
print np.sqrt((np.sum((displacement[2] - displacement2[2]) ** 2))/len(displacement))
print np.sqrt((np.sum((x - (x2/15.0)) ** 2))/len(x))
