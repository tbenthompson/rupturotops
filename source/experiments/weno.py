import scipy.io
import matplotlib.pyplot as pyp
from core.debug import _DEBUG
assert(_DEBUG)
import numpy as np
from numba import autojit
assert(autojit)

import pyweno.weno
class WENO(object):
    """
    Produces an arbitrary order weno reconstruction using the pyWENO
    package.
    """
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


class WENO_NEW2(object):
    """
    Produces arbitrary order weno reconstruction. But, does it by first
    storing the coefficients. Prefered to WENO because then derivatives
    and other operations can be performed on the coefficients.
    """
    def __init__(self, order=5):
        self.order = order
        self.half_width = int((self.order + 1) / 2.0)

    def compute(self, now, direc):
        pass
        # if direc == 1:
        #     direc = 'left'
        # else:
        #     direc = 'right'
        # padded = np.pad(now, self.half_width, 'constant')
        # retval = pyweno.weno.reconstruct(padded, self.order, direc)
        # return retval[self.half_width:-self.half_width]


##############################################################################
# TESTS BELOW HERE
##############################################################################

from core.data_controller import data_root
interactive_test = False

def test_higher_order_weno():
    w = WENO(11)
    assert(w.order == 11)
    assert(w.half_width == 6)
    assert(type(w.half_width) == int)

def _test_new_weno_helper(y, filename):
    weno = WENO()
    left_bdry = weno.compute(y, 1)
    # unidirectional flow, so we only care about incoming from the left
    # and outgoing to the right
    # right_bdry = weno.compute(y, -1)
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
