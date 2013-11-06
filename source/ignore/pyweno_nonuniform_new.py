"""
PyWENO non-uniform reconstruction routines.

Original code written by Matthew Emmett as part of PyWENO.
Edited by Ben Thompson.
"""

import os
import pickle
import cloud

import numpy as np
import sympy
from sympy.utilities.autowrap import autowrap

import pyweno.reconstruction_coeffs as rc
import pyweno.symbolic as symbolic
from core.debug import _DEBUG


#

def reconstruction_coefficients(k, xi, x):
    r"""Numerically compute the reconstruction coefficients for a 2k-1
    order WENO scheme corresponding to the reconstruction points in *xi*
    on the non-uniform grid *x*.

    The reconstruction points in *xi* should be in :math:`[-1, 1]`.  This
    interval is then mapped to the cell :math:`[x_{i-1/2}, x_{i+1/2}]`.

    :param xi: list of reconstruction points
    :param k: reconstruction order
    :param x: cell boundaries

    The returned coefficients are stored in a NumPy array that is
    indexed according to ``c[i,l,r,j]``.  That is

    .. math::

        f^r(\xi_l) \approx \sum_j c^{l,r}_{i,j} \, f_{i-r+j}

    for each :math:`l` from 0 to ``len(xi)``.

    """

    X = np.array(x)
    N = len(X) - 1
    n = len(xi)

    c = np.zeros((N, n, k, k))       # indexed as c[i,l,r,j]

    for i in range(k - 1, N - k + 1):
        for l in range(n):
            z = 0.5 * (X[i] + X[i + 1]) + 0.5 * (X[i + 1] - X[i]) * xi[l]
            for r in range(k):
                c[i, l, r, :] = rc.reconstruction_coeffs(z, i, r, k, X)

    return c


#

def optimal_weights(k, xi, x, tolerance=1e-12):
    r"""Compute the optimal weights for a 2k-1 order WENO scheme
    corresponding to the reconstruction points in *xi* on the
    non-uniform grid *x*.

    The coefficients are stored in a NumPy array that is indexed
    according to ``w[i,l,r]``.  That is

    .. math::

        f(\xi^l) \approx \sum_r w^r f^r

    for each :math:`l` from 0 to ``len(xi)``.
    """

    X = np.array(x)
    n = len(xi)
    N = len(X) - 1

    c = reconstruction_coefficients(k, xi, X)

    # add on a periodic extension...
    X2 = np.zeros(N + 2 * (k) + 1, X.dtype)
    X2[k:-k] = X
    X2[:k] = X[0] - (X[-1] - X[-k - 1:-1])
    X2[-k:] = X[-1] + (X[1:k + 1] - X[0])
    c2k = reconstruction_coefficients(2 * k - 1, xi, X2)

    # chop off the extra bits
    c2k = c2k[k:-k, ...]

    varpi = np.zeros((N, n, k))
    for i in range(k - 1, N - k + 1):
        for l in range(n):

            # we'll just use the first k eqns since the system is
            # overdetermined
            omega = {}
            for j in range(k):

                rmin = max(0, (k - 1) - j)
                rmax = min(k - 1, 2 * (k - 1) - j)

                accum = 0.0
                for r in range(rmin + 1, rmax + 1):
                    accum = accum + omega[r] * c[i, l, r, r - (k - 1) + j]

                omega[rmin] = (c2k[i, l, k - 1, j] - accum) / c[
                    i, l, rmin, rmin - (k - 1) + j]

            # now check all 2*k-1 eqns to make sure the weights work out properly
            # XXX

            for r in range(k):
                varpi[i, l, r] = omega[r]

    return varpi


#
def jiang_shu_file(k, f=None):
    """
    Returns the proper location to save or load
    the Jiang Shu smoothness coefficient functions
    """
    filename_prefix = f
    if filename_prefix is None:
        filename_prefix = 'new_jiang_shu'
    filename_base = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(filename_base, filename_prefix + str(k) + '.pkl')
    return filename


def test_get_filename():
    assert(jiang_shu_file(3, 'abc')
           == os.path.dirname(os.path.abspath(__file__)) + '/abc3.pkl')
    assert(jiang_shu_file(3) == os.path.dirname(
        os.path.abspath(__file__)) + '/new_jiang_shu3.pkl')


def jiang_shu_smoothness_create(k, filename=None):
    """
    Compute the Jiang-Shu smoothness coefficient functions and store them in
    specified file.

    This function uses the sympy autowrap module to create fortran functions
    using f2py. It is massively faster at runtime than directly calling the sympy functions
    using subs. As a tradeoff, this takes a couple minutes to generate the fortran
    functions.
    """

    # the integration variable (represents position within the cell)
    x = sympy.var('x')

    # build array of cell boundaries (sympy vars x[i])
    xs = []
    for j in range(k + 1):
        xs.append(sympy.var('x%d' % j))

    # The upper and lower boundaries of the cell, used as the integration
    # bounds.
    b1 = sympy.var('b1')
    b2 = sympy.var('b2')

    # and build the array of cell averages
    ys = []
    for j in range(k):
        ys.append(sympy.var('y%d' % j))

    p = symbolic.primitive_polynomial_interpolator(xs, ys)
    p = p.as_poly(x)
    p = p.diff(x)

    ppp = {}
    for r in range(0, k):
        # sum of L^2 norms of derivatives
        s = 0
        for j in range(1, k):
            pp = (p.diff((0, j))) ** 2
            pp = pp.integrate(x)
            pp = (b2 - b1) ** (2 * j - 1) * (
                pp.subs(x, b2) - pp.subs(x, b1))
            s = s + pp

        # This is slow but, because of some bug in sympy, the 'coeff'
        # function doesn't work properly otherwise?!
        ppp[r] = s.simplify().expand()

    # Take the smoothness expressions and make a fortran function out
    # of each coefficient. Pickle these functions to the specified file
    filename = jiang_shu_file(k, filename)
    fncs = []
    for r in range(0, k):
        coeffs_list = []
        for i in range(0, k):
            inner_coeffs_list = []
            for j in range(0, k):
                args = [xs[m] for m in range(0, k + 1)]
                args.append(b1)
                args.append(b2)
                coeffs_expr = ppp[r].coeff(ys[i] * ys[j])
                _DEBUG()
                # to finish this implementation, i would need to use the sympy codegen utility and
                # f2py to create fortran functions for the smoothness coefficients. write the
                # code here
                # l = autowrap([coeffs_expr], args=args, tempdir=autogen_fortran/
                inner_coeffs_list.append(l)
            coeffs_list.append(inner_coeffs_list)
        fncs.append(coeffs_list)
    _DEBUG(2)
    with open(filename, 'w') as f:
        cloud.serialization.cloudpickle.dump(fncs, f)

def generate_function(coeffs_expr, args, directory, name):

# This test takes forever to run, so it's okay to turn it off unless it's
# needed.
def test_jiang_shu_smoothness_create():
    jiang_shu_smoothness_create(3)

def jiang_shu_smoothness_coefficients(k, x, filename=None):
    r"""Compute the Jiang-Shu smoothness coefficients for a 2k-1 order
    WENO scheme on the non-uniform grid *x*.

    The coefficients are stored in a NumPy array indexed according to
    ``beta[i,r,m,n]``.  That is

    .. math::

        \sigma^r = \sum_{m=1}^{2k-1} \sum_{n=1}^{2k-1}
            \beta_{r,m,n}\, \overline{f}_{i-k+m}\, \overline{f}_{i-k+n}.
    """

    X = np.array(x)
    N = len(X) - 1

    filename = jiang_shu_file(k, filename)
    with open(filename, 'r') as f:
        fncs = pickle.load(f)

    # compute reconstruction coefficients for each left shift r
    beta = np.zeros((N, k, k, k))
    for i in range(k - 1, N - k + 1):
        for r in range(0, k):
            args = []
            for j in range(0, k + 1):
                args.append([X[i - r + j]])
            args.append([X[i]])
            args.append([X[i + 1]])
            # pick out coefficients
            for m in range(k):
                for n in range(m, k):
                    c = fncs[r][m][n](*args)
                    if c is not None:
                        beta[i, r, m, n] = float(c)

    return beta

def test_jiang_shu_smoothness_coefficients():
    correct = \
        [[[  3.33333333, -10.33333333,   3.66666667],
         [  0.        ,   8.33333333,  -6.33333333],
         [  0.        ,   0.        ,   1.33333333]],
        [[  1.33333333,  -4.33333333,   1.66666667],
         [  0.        ,   4.33333333,  -4.33333333],
         [  0.        ,   0.        ,   1.33333333]],
        [[  1.33333333,  -6.33333333,   3.66666667],
         [  0.        ,   8.33333333, -10.33333333],
         [  0.        ,   0.        ,   3.33333333]]]
    # Check for the correct answer on a uniform grid.
    beta = jiang_shu_smoothness_coefficients(3, [1, 2, 3, 4, 5, 6, 7])
    assert(np.testing.assert_almost_equal(beta[3], correct))

    # Make sure it doesn't fail in the case of a non-uniform grid.
    # How to check correctness?
    beta2 = jiang_shu_smoothness_coefficients(3, [1, 2, 3, 4, 5, 6, 7])
    assert(beta2 is not None)
