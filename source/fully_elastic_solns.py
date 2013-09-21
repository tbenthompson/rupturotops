from math import pi,factorial   
import numpy as np
import scipy.integrate as integrate
from matplotlib import pyplot as pyp

from utilities import A_integral

#derived by differentiating the slip for slips delta z apart
def elastic_half_space(slip_distribution, x):
    return (x/pi) * A_integral(slip_distribution, 1, x, 0)

#based on the infinite images solution
def two_layer_elastic(slip_distribution, alpha, beta, x, truncation = 10):
    outer_factor = x/pi
    main_fault_term = A_integral(slip_distribution, alpha, x, 0)
    images = 0
    for i in range(1, truncation + 1):
        comp1 = A_integral(slip_distribution, alpha, x, 2 * i) 
        comp2 = A_integral(slip_distribution, alpha, x, -2 * i)
        images += (beta ** i) * (comp1 + comp2)
    result = outer_factor * (main_fault_term + images)
    return result


