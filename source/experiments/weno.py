"""
This functionality is currently on indefinite hold in
favor of the TVD flux implementation of ADER schemes
"""
import scipy.io
import matplotlib.pyplot as pyp
from core.debug import _DEBUG
import numpy as np

import pyweno.weno
from pyweno.nonuniform import reconstruction_coefficients, optimal_weights
from pyweno.nonuniform import jiang_shu_smoothness_coefficients
import pyximport
pyximport.install()
import experiments.mult_with_smoothness


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

    Uses the pyWENO nonuniform interface.
    """

    def __init__(self, mesh, order=5):
        self.mesh = mesh
        self.order = order
        self.half_width = int((self.order + 1) / 2.0)
        padded_edges = self.mesh.extend_edges(self.half_width + 1)
        self.coeffs = reconstruction_coefficients(3, [-1.0, 1.0],
                                                  padded_edges)
        self.coeffs = self.coeffs[self.half_width - 1:-self.half_width + 1]
        self.weights = optimal_weights(3, [-1.0, 1.0], padded_edges)
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

        padded = np.pad(now, self.half_width - 1, 'constant')
        return compute_helper(now, padded, self.half_width, side_index, self.coeffs,
                              self.smoothness, self.weights, self.eps)

def compute_helper(now, padded, half_width, side_index, coeffs, smoothness, weights, eps):
    cells = len(now)
    weights = weights[:, side_index]
    coeffs = coeffs[:, side_index]
    beta = experiments.mult_with_smoothness.mult_with_smoothness(cells, half_width, padded, smoothness)
    s_weights = scaled_weights(cells, half_width, eps, weights, beta)
    small_polynomials = mult_with_coeffs(cells, half_width, padded, coeffs)
    reconstruction = mult_polynomials(cells, half_width, s_weights, small_polynomials)
    return np.array(reconstruction)

def mult_polynomials(cells, half_width, weights, small_polynomials):
    reconstruction = [0.0 for row in range(cells)]
    for i in range(cells):
        for j in range(half_width):
            reconstruction[i] += weights[i][j] * small_polynomials[i][j]
    return reconstruction

def scaled_weights(cells, half_width, eps, weights, beta):
    scaled = [[0.0 for col in range(half_width)] for row in range(cells)]
    sum = [0.0 for row in range(cells)]
    for i in range(cells):
        for j in range(half_width):
            scaled[i][half_width - 1 - j] = weights[i][j] / \
                ((eps + beta[i][half_width - 1 - j]) ** 2)
            sum[i] += scaled[i][half_width - 1 - j]
        if sum[i] == 0.0:
            continue
        for j in range(half_width):
            scaled[i][j] /= sum[i]
    return scaled

def mult_with_coeffs(cells, half_width,  padded, coeffs):
    retval = [[0.0 for col in range(half_width)] for row in range(cells)]
    for i in range(cells):
        center = half_width - 1 + i
        for r in range(half_width):
            #pyWENO does a weird flipping with its smoothness coefficients
            #so we go backwards
            for j in range(half_width):
                #note that this result is in the backwards pyWENO form
                #retval = [poly3, poly2, poly1] in the Shu 2009 notation
                retval[i][half_width - 1 - r] += coeffs[i][r][j] *\
                    padded[center - r + j]
    return retval


##############################################################################
# TESTS BELOW HERE
##############################################################################
from core.data_controller import data_root
from core.mesh import Mesh
interactive_test = False

def test_weight_scaling():
    weights = np.array([[0.1, 0.6, 0.3]])
    beta = np.array([[1.0, 2.0, 1.0]])
    correct = np.array([[0.3/0.55, 0.15/0.55, 0.1/0.55]])
    result = scaled_weights(1, 3,  1e-6, weights, beta)
    np.testing.assert_almost_equal(np.sum(result), 1.0)
    np.testing.assert_almost_equal(result, correct, 5)


def test_mult_with_smoothness():
    # test with the uniform smoothness coefficients in
    # Shu 2009
    now_chunk = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    smoothness = \
        [[[[  3.33333333, -10.33333333,   3.66666667],
         [  0.        ,   8.33333333,  -6.33333333],
         [  0.        ,   0.        ,   1.33333333]],
        [[  1.33333333,  -4.33333333,   1.66666667],
         [  0.        ,   4.33333333,  -4.33333333],
         [  0.        ,   0.        ,   1.33333333]],
        [[  1.33333333,  -6.33333333,   3.66666667],
         [  0.        ,   8.33333333, -10.33333333],
         [  0.        ,   0.        ,   3.33333333]]]]
    correct = \
        13.0 / 12.0 * np.array([[0.0, 0.0, 1.0]]) ** 2 + \
        1.0 / 4.0 * np.array([[2.0, -2.0, -1.0]]) ** 2
    beta = mult_with_smoothness(1, 3, now_chunk, smoothness)
    np.testing.assert_almost_equal(beta, correct)


def test_mult_with_coeffs():
    now = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    coeffs = np.array(
        [[[ 0.33333333,  0.83333333, -0.16666667],
         [-0.16666667,  0.83333333,  0.33333333],
         [ 0.33333333, -1.16666667,  1.83333333]]])
    correct = np.array([[21.0/6.0, 21.0/6.0, 20.0/6.0]])
    result = mult_with_coeffs(1, 3, now, coeffs)
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
    np.testing.assert_almost_equal(y_exact[0], y)
    if interactive_test:
        pyp.plot(dx.T)
        pyp.plot(comp['z'].T)
        pyp.show()
    np.testing.assert_almost_equal(comp['z'][0], dx)


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
