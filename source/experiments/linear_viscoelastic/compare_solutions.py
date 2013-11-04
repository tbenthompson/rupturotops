import numpy as np
from math import pi
import unittest
import full_elastic
import constant_slip_maxwell_dimensionless
import constant_slip_maxwell
import utilities
import full_viscoelastic


class CompareSolutions(unittest.TestCase):
    # compares the two layers elastic solution to the one layer solution in the
    # case that the moduli are equal -- the degenerate case
    def testCompareElasticSolutions(self):
        x = np.arange(0.05, 5.0, 0.05)
        displacement = full_elastic.elastic_half_space(lambda z: 1.0, x)
        displacement2 = full_elastic.two_layer_elastic(lambda z: 1.0,
                                                       1.0, 0.0, x)
        self.assertTrue(utilities.mse(displacement, displacement2) < 0.0001)

    # this compares the dimensional segall solution to my dimensionless
    # solution using my parameters
    def testCompareViscoelasticSolutions(self):
        alpha = 1.0

        u_benchmark, x2, H = self.\
            testCompareSegallViscoelasticSolutionWithSimpleAnalytic()
        x = x2 / H
        t = np.arange(0.0, 5.0)
        u_estimate = map(lambda t_in:
                         constant_slip_maxwell_dimensionless.
                         solution(x, t_in, alpha), t)

        for i in range(len(t)):
            self.assertTrue(utilities.mse(u_estimate[i], u_benchmark[i])
                            < 0.01)
        # utilities.plot_time_series_1D(x, u_estimate, t)

    #check that this solution devolves to the elastic solution at t = 0
    #the arctan solution at time t = 0
    def testCompareSegallViscoelasticSolutionWithSimpleAnalytic(self):
        slip = 1.0
        depth_of_fault = 15.
        H = 15.
        x = np.arange(-100.01, 100.01)
        t_over_t_r = np.arange(0.0, 5.0)
        u = map(lambda t: constant_slip_maxwell.solution(x, t, depth_of_fault, H, slip), t_over_t_r)

        time0analyticsolution = (slip / pi) * np.arctan(depth_of_fault / x)
        self.assertTrue(utilities.mse(u[0], time0analyticsolution) < 0.0001)
        return u, x, H

    def testCompareVariableSlip(self):
        slip = lambda z: 1
        x = np.arange(0.05, 10.0, 0.5)
        t = np.arange(0.0, 5.0)
        alpha = 1

        u_t = map(lambda t_in: full_viscoelastic.solution(slip, x, t_in, alpha), t)

        #first compare to elastic solutions -- they should match at t=0
        u_e = full_elastic.elastic_half_space(slip, x)
        # print utilities.mse(u_t[0], u_elastic)
        self.assertTrue(utilities.mse(u_t[0], u_e) < 0.0001)

        #then compare to viscoelastic solutions -- they should match at all times because slip is constant
        u_constant_slip_ve = map(lambda t_in: constant_slip_maxwell_dimensionless.solution(x, t_in, alpha), t)
        # pyp.figure(1)
        # utilities.plot_time_series_1D(x, u_t, t, show = False)
        # pyp.figure(2)
        # utilities.plot_time_series_1D(x, u_constant_slip_ve, t, show = False)
        # pyp.show()
        for i in range(len(t)):
            # print u_t[i] - u_constant_slip_ve[i]
            self.assertTrue(utilities.mse(u_t[i], u_constant_slip_ve[i]) < 0.0001)

    def testAlphaLimit(self):
        slip = lambda z: 1
        x = np.arange(0.05, 10.0, 0.50)
        t = np.arange(0.0, 5.0)
        alpha = 1000000000.0
        u_t = np.array(map(lambda t_in: full_viscoelastic.solution(slip, x, t_in, alpha), t))
        self.assertTrue((u_t < 0.000000001).all())

    def testXLimit(self):
        slip = lambda z: 1
        x = np.array([0.001, 100000000])
        t = [0]
        alpha = 1.0
        u_t = np.array(map(lambda t_in: full_viscoelastic.solution(slip, x, t_in, alpha), t))
        self.assertTrue((abs(u_t[0][0] - 0.5) < 0.001).all())
        self.assertTrue((abs(u_t[0][1] - 0.0) < 0.001).all())

    def testSlipLimit(self):
        slip = lambda z: 0
        x = np.arange(0.05, 10.0, 0.05)
        t = [0]
        alpha = 1.0
        u_t = np.array(map(lambda t_in: full_viscoelastic.solution(slip, x, t_in, alpha), t))
        self.assertTrue((u_t == 0.0).all())

    def testmse(self):
        a = utilities.mse([1,2,3],[0,1,2])
        self.assertTrue(a == 1)

    def testA_Integral(self):
        a = utilities.A_integral(lambda z: 1, [1], 1, 0)
        #arctan(1) - arctan(0) = pi/4..... arctan(0) = 0
        self.assertTrue(abs(a - (pi/4)) < 0.0001)

        b = utilities.A_integral(lambda z: 2, [1], 1, 0)
        self.assertTrue(abs(b - (pi/2)) < 0.0001)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CompareSolutions))
    unittest.TextTestRunner(verbosity=2).run(suite)
