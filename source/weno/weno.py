"""
This functionality is currently on indefinite hold in
favor of the TVD flux implementation of ADER schemes
"""
import scipy.io
import matplotlib.pyplot as pyp
from core.debug import _DEBUG
import numpy as np
# from numba import autojit

import pyweno.weno
from pyweno_nonuniform import reconstruction_coefficients, optimal_weights
from pyweno_nonuniform import jiang_shu_smoothness_coefficients


class WENO(object):

    """
    Produces an arbitrary order weno reconstruction using the pyWENO
    package. Only works properly for uniform mesh spacings.
    """

    def __init__(self, mesh, order=5):
        self.order = order
        self.half_width = int((self.order + 1) / 2.0)

    def compute(self, now, direc):
        padded = np.pad(now, self.half_width, 'constant')
        retval = pyweno.weno.reconstruct(padded, self.order, direc)
        return retval[self.half_width:-self.half_width]


class WENO_NEW2(object):

    """
    Produces arbitrary order weno reconstruction. But, does it by first
    storing the coefficients. Prefered to WENO because then derivatives
    and other operations can be performed on the coefficients.

    This should be migrated to using sympy variables so that the expressions
    can be differentiated and used for the generalized Riemann problem
    in the ADER scheme. In fact, it should probably just be hardcoded for
    order = 5. Or I could implement some saving and loading scheme?

    SUPER SLOW IMPLEMENTATION!!
    """

    def __init__(self, mesh, order=5):
        self.mesh = mesh
        self.order = order
        self.half_width = int((self.order + 1) / 2.0)
        padded_edges = self.mesh.extend_edges(self.half_width + 1)
        self.coeffs = reconstruction_coefficients(3, [-1, 1],
                                                  padded_edges)
        self.coeffs = self.coeffs[self.half_width - 1:-self.half_width + 1]
        self.weights = optimal_weights(3, [-1, 1], padded_edges)
        self.weights = self.weights[self.half_width - 1:-self.half_width + 1]
        self.smoothness = jiang_shu_smoothness_coefficients(3,
                                                            padded_edges)
        self.smoothness = self.smoothness[self.half_width - 1:
                                          -self.half_width + 1]
        self.eps = 1e-6

    def compute(self, now, side):
        if side == 'left':
            side_index = 0
        else: #side == 'right'
            side_index = 1

        result = []
        padded = np.pad(now, self.half_width - 1, 'constant')
        # test_range = range(side_index, len(now) - 1 + side_index)
        for i in range(0, len(now)):
            which_chunk = self.get_chunk(padded, self.half_width - 1 + i)
            which_coeffs = self.coeffs[i][side_index]
            which_smoothness = self.smoothness[i]
            which_weights = self.weights[i][side_index][::-1]
            beta = self.mult_with_smoothness(which_chunk, which_smoothness)
            weights = self.scaled_weights(which_weights, beta)
            small_polynomials = self.mult_with_coeffs(which_chunk,
                                                      which_coeffs)
            result.append(np.sum(weights * small_polynomials))
        # _DEBUG(10)
        return np.array(result)

    def get_chunk(self, padded, center, reverse=False):
        retval = []
        for i in range(-self.half_width + 1, self.half_width):
            retval.append(padded[center + i])
        return np.array(retval)

    def mult_with_coeffs(self, now_chunk, coeffs):
        retval = []
        for r in range(0, self.half_width):
            #pyWENO does a weird flipping with its smoothness coefficients
            #so we go backwards
            lower = (self.half_width - 1 - r)
            upper = (self.half_width - 1 - r) + self.half_width
            retval.append(np.sum(coeffs[r] *
                          now_chunk[lower:upper]))
        #note that this result is in the backwards pyWENO form
        #retval = [poly3, poly2, poly1] in the Shu 2009 notation
        return np.array(retval)[::-1]

    def mult_with_smoothness(self, now_chunk, smoothness):
        retval = []
        for r in range(0, self.half_width):
            #pyWENO does a weird flipping with its smoothness coefficients
            #so we go backwards
            lower = (self.half_width - 1 - r)
            upper = (self.half_width - 1 - r) + self.half_width
            red_chunk = now_chunk[lower:upper]
            retval.append(np.sum(smoothness[r] * \
                                 np.outer(red_chunk, red_chunk)))
        #note that this result is in the backwards pyWENO form
        #retval = [beta3, beta2, beta1] in the Shu 2009 notation
        return np.array(retval)[::-1]

    def scaled_weights(self, weights, beta):
        w = weights / ((self.eps + beta) ** 2)
        sum_w = np.sum(w)
        if sum_w == 0.0:
            return w
        w = w / sum_w
        return w


##############################################################################
# TESTS BELOW HERE
##############################################################################
from core.data_controller import data_root
from core.mesh import Mesh
interactive_test = False

def test_get_chunk():
    m = Mesh()
    w = WENO_NEW2(m)
    padded = np.array([0, 0, 1, 1, 1, 2, 2, 0.5, 0, 0])
    correct = np.array([1, 2, 2, 0.5, 0])
    assert((w.get_chunk(padded, 6) == correct).all())

def test_weight_scaling():
    m = Mesh()
    w = WENO_NEW2(m)
    weights = np.array([0.1, 0.6, 0.3])
    beta = np.array([1.0, 2.0, 1.0])
    correct = np.array([0.1/0.55, 0.15/0.55, 0.3/0.55])
    result = w.scaled_weights(weights, beta)
    np.testing.assert_almost_equal(np.sum(result), 1.0)
    np.testing.assert_almost_equal(result, correct, 5)


def test_mult_with_smoothness():
    # test with the uniform smoothness coefficients in
    # Shu 2009
    m = Mesh()
    w = WENO_NEW2(m)
    now_chunk = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    smoothness = \
        [[[  3.33333333, -10.33333333,   3.66666667],
         [  0.        ,   8.33333333,  -6.33333333],
         [  0.        ,   0.        ,   1.33333333]],
        [[  1.33333333,  -4.33333333,   1.66666667],
         [  0.        ,   4.33333333,  -4.33333333],
         [  0.        ,   0.        ,   1.33333333]],
        [[  1.33333333,  -6.33333333,   3.66666667],
         [  0.        ,   8.33333333, -10.33333333],
         [  0.        ,   0.        ,   3.33333333]]]
    correct = \
        13.0 / 12.0 * np.array([0.0, 0.0, 1.0]) ** 2 + \
        1.0 / 4.0 * np.array([2.0, -2.0, -1.0]) ** 2
    beta = w.mult_with_smoothness(now_chunk, smoothness)
    assert(type(beta) == np.ndarray)
    np.testing.assert_almost_equal(beta, correct)


def test_mult_with_coeffs():
    m = Mesh()
    w = WENO_NEW2(m)
    now_chunk = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    coeffs = np.array(
        [[ 0.33333333,  0.83333333, -0.16666667],
         [-0.16666667,  0.83333333,  0.33333333],
         [ 0.33333333, -1.16666667,  1.83333333]])
    correct = np.array([21.0/6.0, 21.0/6.0, 20.0/6.0])
    result = w.mult_with_coeffs(now_chunk, coeffs)
    assert(type(result) == np.ndarray)
    np.testing.assert_almost_equal(result, correct)


def test_weno_compare_hard():
    y_harder = np.array([0.0, 0.0, 40.0, 0.0, 0.0])
    _test_weno_compare_helper(y_harder)

def test_weno_compare_easy():
    y_easy = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    _test_weno_compare_helper(y_easy)

def _test_weno_compare_helper(y):
    dx = np.ones(len(y))
    m = Mesh(dx)
    w1 = WENO(m)
    w2 = WENO_NEW2(m)
    result1 = w1.compute(y, 'left')
    result2 = w2.compute(y, 'left')
    if interactive_test:
        pyp.plot(result1)
        pyp.plot(result2)
        pyp.show()
    np.testing.assert_almost_equal(result1, result2)
    result1 = w1.compute(y, 'right')
    result2 = w2.compute(y, 'right')
    np.testing.assert_almost_equal(result1, result2)

def test_nonuniform_weno():
    dx = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    y = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    m = Mesh(dx)
    w1 = WENO(m)
    w2 = WENO_NEW2(m)
    result1 = w1.compute(y, 'left')
    result2 = w2.compute(y, 'left')
    assert((result1 != result2).all())

def test_higher_order_weno():
    w = WENO(Mesh(), 11)
    assert(w.order == 11)
    assert(w.half_width == 6)
    assert(type(w.half_width) == int)


def _test_weno_helper(x, y, filename):
    weno = WENO_NEW2(Mesh((x - np.roll(x, 1))[1:]))
    left_bdry = weno.compute(y, 'left')
    # unidirectional flow, so we only care about incoming from the left
    # and outgoing to the right
    # right_bdry = weno.compute(y, -1)
    # test assumes all flow is to the right
    dx = -(left_bdry - np.roll(left_bdry, -1))
    comp = scipy.io.loadmat(filename)
    y_exact = comp['fo']
    assert(np.all(abs(y_exact - y) < 0.000000001))
    if interactive_test:
        pyp.plot(dx.T)
        pyp.plot(comp['z'].T)
        pyp.show()
    assert(np.all(abs(comp['z'] - dx) < 0.0001))


def test_weno_more():
    x = np.linspace(-5, 5, 100)
    y = np.where(x <= 0.0, np.ones_like(x), np.zeros_like(x))
    filename = data_root + '/test/test_weno_jumpy.mat'
    _test_weno_helper(x, y, filename)


def test_weno_smooth():
    x = np.linspace(-5, 5, 100)
    y = np.exp(-x ** 2)
    filename = data_root + '/test/test_weno_smooth.mat'
    _test_weno_helper(x, y, filename)