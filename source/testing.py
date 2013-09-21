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
        self.assertTrue(mse(displacement,displacement2) < 0.01)
    
    #this compares the dimensional segall solution to my dimensionless solution using my parameters
    def testCompareViscoelasticSolutions(self):
        alpha = 1.0
        a = -1.0

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

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSolutions))
    unittest.TextTestRunner(verbosity=2).run(suite)
