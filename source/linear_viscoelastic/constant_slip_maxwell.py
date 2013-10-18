from math import pi,factorial   
import numpy as np

#displacements

#all the functions for the maxwell viscoelastic solution in segall
def _F(x, D, H, n):
    retval = np.arctan((2*x*D)/(x**2 + (2*n*H)**2 - D**2))
    # print retval
    return retval

def _E(time_scale,n):
    retval = 0
    for j in range(1, n+1):
        # print str(j) + " " + str(n)
        # print time_scale
        retval += ((time_scale)**(n-j))/factorial(n-j)
    return retval

def solution(x, time_scale, depth_of_fault, H, slip):
    u_3 = 0
    for i in range(20):
        u_3 += (1 - (np.exp(-time_scale) * _E(time_scale,i+1))) * _F(x, depth_of_fault, H, i+1)
    u_3 += np.arctan(depth_of_fault/x)
    u_3 *= slip/pi
    return u_3
