"""
PyWENO non-uniform reconstruction routines.

Original code written by Matthew Emmett as part of PyWENO.
Edited by Ben Thompson.
"""
import numpy as np
from core.debug import _DEBUG
import weno_code_gen as wcg


def call_coeffs(k, d, j, args)


def reconstruction_coefficients(k, xi, x, d=0):
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
            # Map the point under consideration back into original
            # spatial domain
            z = 0.5 * (X[i] + X[i + 1]) +\
                0.5 * (X[i + 1] - X[i]) * xi[l]
            for r in range(k):
                args = []
                args
                fnc =
                c[i, l, r, :] = rc.reconstruction_coeffs(z, i, r, k, X)


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

            # now check all 2*k-1 eqns to make sure the weights
            # work out properly
            for r in range(k):
                varpi[i, l, r] = omega[r]

    return varpi


def call_smoothness(module, k, r, i, j, args):
    fnc_name = wcg.smoothness_fnc_name(k, r, i, j)
    return wenok.__dict__[fnc_name](*args)


def jiang_shu_smoothness_coefficients(k, x):
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
                    c = call_smoothness(wenok, k, r, m, n, args)
                    if c is not None:
                        beta[i, r, m, n] = float(c)
    return beta

wenok = __import__('weno_3')


def test_jiang_shu_smoothness_coefficients():
    correct = np.array(
        [[[3.33333333, -10.33333333,   3.66666667],
         [0.,   8.33333333,  -6.33333333],
         [0.,   0.,   1.33333333]],
         [[1.33333333,  -4.33333333,   1.66666667],
          [0.,   4.33333333,  -4.33333333],
          [0.,   0.,   1.33333333]],
         [[1.33333333,  -6.33333333,   3.66666667],
          [0.,   8.33333333, -10.33333333],
          [0.,   0.,   3.33333333]]])
    # Check for the correct answer on a uniform grid.
    beta = jiang_shu_smoothness_coefficients(3, [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_almost_equal(beta[3], correct)

    # Make sure it doesn't fail in the case of a non-uniform grid.
    # How to check correctness?
    beta2 = jiang_shu_smoothness_coefficients(3, [1, 2, 3, 3.5, 5, 6, 7])
    assert(beta2 is not None)
