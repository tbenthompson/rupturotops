import numpy as np
from math import pi
import unittest

import fully_elastic_solns 
import segall_maxwell_dimensionless 
import segall_maxwell
import utilities


class TestSolutions(unittest.TestCase):    
    #compares the two layers elastic solution to the one layer solution in the case that the moduli are equal -- the degenerate case
    def testCompareElasticSolutions(self):
        displacement = []
        x = np.arange(-5.0,5.0,0.05)
        displacement2 = []
        for i in range(len(x)):
            displacement.append(fully_elastic_solns.elastic_half_space(lambda z: 1.0, x[i]))
            displacement2.append(fully_elastic_solns.two_layer_elastic(lambda z: 1.0, 1.0, 0.0, x[i]))
        self.assertTrue(utilities.mse(displacement,displacement2) < 0.01)
    
    #this compares the dimensional segall solution to my dimensionless solution using my parameters
    def testCompareViscoelasticSolutions(self):
        alpha = 1.0
        a = -1.0

        u_benchmark, x2, H = self.testCompareSegallViscoelasticSolutionWithSimpleAnalytic()
        x = x2/H
        t = np.arange(0.0,5.0)

        u_estimate = []
        for i in range(len(t)):
            u_estimate.append([])
            for j in range(len(x)):
                u_estimate[i].append(segall_maxwell_dimensionless.constant_slip_constant_shear_modulus_viscoelastic(x[j], t[i], alpha, a))
        
        for i in range(len(t)):
            self.assertTrue(utilities.mse(u_estimate[i], u_benchmark[i]) < 0.01)
        # utilities.plot_time_series_1D(x, u_estimate, t)

    def testCompareSegallViscoelasticSolutionWithSimpleAnalytic(self):
        slip = 1.0
        depth_of_fault = 15.
        H = 15.
        x = np.arange(-100.01, 100.01)
        t_over_t_r = np.arange(0.0, 5.0)
        u = map(lambda t: segall_maxwell.calc_disp(x, t, depth_of_fault, H, slip), t_over_t_r)

        #check that this solution devolves to the elastic solution at t = 0
        #the arctan solution at time t = 0
        time0analyticsolution = (slip / pi) * np.arctan(depth_of_fault / x)
        self.assertTrue(utilities.mse(u[0], time0analyticsolution) < 0.01) 
        return u, x, H

    def testCompareFullWithSegallMaxwellDimensionless(self):
        slip_distribution = lambda z: 1
        x = np.arange(-10.0, 10.0, 0.2)
        t = np.arange(0.0, 5.0)
        alpha = 1
        beta =  

    def testAIntegral(self):
        a = utilities.A_integral(lambda z: 1, 1, 1, 0)
        #arctan(1) - arctan(0) = pi/4..... arctan(0) = 0
        self.assertTrue(abs(a - (pi/4)) < 0.0001)

        b = utilities.A_integral(lambda z: 2, 1, 1, 0)
        self.assertTrue(abs(b - (pi/2)) < 0.0001)

    #compare the solution with variation in elastic modulus with the other solutions
    def testCompareVariableModuliSolution(self):
        pass        


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSolutions))
    unittest.TextTestRunner(verbosity=2).run(suite)
