import scipy.io
import matplotlib.pyplot as pyp
from core.debug import _DEBUG
assert(_DEBUG)
import numpy as np
from numba import autojit
assert(autojit)
# This should be migrated to use PyWENO and construct a matrix
# vector product which can be optimized in PETSc


class WENO(object):
    """
    Produces a 5th order WENO reconstruction per Jiang and Shu (1996)
    """
    def __init__(self):
        self.eps = 1e-6
        self.g = np.array([1.0 / 10.0, 3.0 / 5.0, 3.0 / 10.0])

    def compute(self, now, direc):
        #these variables represent the left-two, left-one, central,
        #right and right-two average cell values. where left and
        #right are flipped if we're looking at a left boundary
        # where direc = -1 (note that a leftwards value requires
        # a right roll, thus np.roll(now, 1) gives the leftwards
        # values with the same index as center value
        l2 = np.roll(now, -2 * direc)
        l = np.roll(now, -direc)
        c = np.roll(now, 0)
        r = np.roll(now, direc)
        r2 = np.roll(now, 2 * direc)

        terms = self.get_terms(l2, l, c, r, r2)
        betas = self.get_beta(l2, l, c, r, r2)
        weights = self.get_weights(betas)

        flux = np.sum(weights * terms, 0)
        return flux

    #@autojit()
    def get_terms(self, l2, l, c, r, r2):
        terms = []
        terms.append((2.0 * r2 - 7.0 * r + 11.0 * c) / 6.0)
        terms.append((-r + 5.0 * c + 2.0 * l) / 6.0)
        terms.append((2.0 * c + 5.0 * l - l2) / 6.0)
        return np.array(terms)

    #@autojit()
    def get_beta(self, l2, l, c, r, r2):
        cd_l = l2 - 2.0 * l + c
        cd_c = l - 2.0 * c + r
        cd_r = c - 2.0 * r + r2
        betas = []
        betas.append((13.0 / 12.0) * cd_r ** 2.0 +
                     (1.0 / 4.0) * (r2 - 4.0 * r + 3.0 * c) ** 2.0)
        betas.append((13.0 / 12.0) * cd_c ** 2.0
                     + (1.0 / 4.0) * (r - l) ** 2.0)
        betas.append((13.0 / 12.0) * cd_l ** 2.0 +
                     (1.0 / 4.0) * (3.0 * c - 4.0 * l + l2) ** 2.0)
        return np.array(betas)

    #@autojit()
    def get_weights(self, betas):
        weights = self.g * (1 / ((self.eps + betas.T) ** 2))
        weights = weights.T / np.sum(weights, 1)
        return weights
        # pyp.plot(weight1)

import pyweno.weno
class WENO_NEW(object):
    def __init__(self, order=5):
        self.order = order
        self.half_width = int((self.order + 1) / 2.0)

    def compute(self, now, direc):
        if direc == 1:
            direc = 'left'
        else:
            direc = 'right'
        padded = np.pad(now, self.half_width, 'constant')
        retval = pyweno.weno.reconstruct(padded, self.order, direc)
        return retval[self.half_width:-self.half_width]




##############################################################################
# TESTS BELOW HERE
##############################################################################

from core.data_controller import data_root
interactive_test = False

def test_higher_order_weno():
    w = WENO_NEW(11)
    assert(w.order == 11)
    assert(w.half_width == 6)
    assert(type(w.half_width) == int)

def test_compare_wenos():
    x = np.linspace(-5, 5, 100)
    y = np.where(x <= 0.0, np.ones_like(x), np.zeros_like(x))
    one = WENO_NEW()
    two = WENO()
    one_left = one.compute(y, 1)
    one_right = one.compute(y, -1)
    two_left = two.compute(y, 1)
    two_right = two.compute(y, -1)
    np.testing.assert_almost_equal(one_left, two_left)
    np.testing.assert_almost_equal(one_right, two_right)

def _test_new_weno_helper(y, filename):
    weno = WENO_NEW()
    left_bdry = weno.compute(y, 1)
    right_bdry = weno.compute(y, -1)
    #test assumes all flow is to the right
    dx = -(left_bdry - np.roll(left_bdry, -1))
    comp = scipy.io.loadmat(filename)
    y_exact = comp['fo']
    assert(np.all(abs(y_exact - y) < 0.000000001))
    assert(np.all(abs(comp['z'] - dx) < 0.0000000001))
    if interactive_test:
        pyp.plot(dx.T)
        pyp.plot(comp['z'].T)
        pyp.show()

def test_weno_more():
    x = np.linspace(-5, 5, 100)
    y = np.where(x <= 0.0, np.ones_like(x), np.zeros_like(x))
    filename = data_root + '/test/test_weno_jumpy.mat'
    _test_new_weno_helper(y, filename)

def test_weno_smooth():
    x = np.linspace(-5, 5, 100)
    y = np.exp(-x ** 2)
    filename = data_root + '/test/test_weno_smooth.mat'
    _test_new_weno_helper(y, filename)

def _test_weno_helper(y, filename):
    """
    This test loads 1D data produced by an established, correctly designed
    5th order WENO routine and compares it to the output produced by this
    code.
    """
    weno = WENO()

    #We look at the right boundary of
    direc = -1
    l2 = np.roll(y, -2 * direc)
    l = np.roll(y, -direc)
    c = np.roll(y, 0)
    r = np.roll(y, direc)
    r2 = np.roll(y, 2 * direc)
    terms = weno.get_terms(l2, l, c, r, r2)
    betas = weno.get_beta(l2, l, c, r, r2)

    wts = weno.get_weights(betas)
    flux = weno.compute(y, direc)
    dx = -(flux - np.roll(flux, -1))

    comp = scipy.io.loadmat(filename)
    y_exact = comp['fo']
    #Check that the final result AND
    #all the weights are the same
    assert(np.all(abs(y_exact - y) < 0.000000001))
    assert(np.all(abs(betas[0] - comp['b1']) < 0.00000001))
    assert(np.all(abs(betas[1] - comp['b2']) < 0.00000001))
    assert(np.all(abs(betas[2] - comp['b3']) < 0.00000001))
    assert(np.all(abs(terms[0] - comp['fp1']) < 0.00000001))
    assert(np.all(abs(terms[0] - comp['fp1']) < 0.00000001))
    assert(np.all(abs(terms[1] - comp['fp2']) < 0.00000001))
    assert(np.all(abs(terms[2] - comp['fp3']) < 0.00000001))
    assert(np.all(abs(wts[0] - comp['w1']) < 0.000000001))
    assert(np.all(abs(wts[1] - comp['w2']) < 0.000000001))
    assert(np.all(abs(wts[2] - comp['w3']) < 0.000000001))
    assert(np.all(abs(comp['z'] - dx) < 0.0000000001))
    #Change the below to True to see the comparison
    #if it doesn't match
    if False:
        pyp.plot(comp['z'].T)
        pyp.plot(dx.T)
        pyp.show()

    # assert((diff == 0).all())
