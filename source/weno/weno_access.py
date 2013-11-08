"""
PyWENO non-uniform reconstruction routines.

Original code written by Matthew Emmett as part of PyWENO.
Edited by Ben Thompson.
"""
import numpy as np
import pickle
from weno_code_gen import jiang_shu_file

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

    filename = jiang_shu_file(k, 'smoothness', filename)
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
        [[[3.33333333, -10.33333333,   3.66666667],
         [0.,   8.33333333,  -6.33333333],
         [0.,   0.,   1.33333333]],
         [[1.33333333,  -4.33333333,   1.66666667],
          [0.,   4.33333333,  -4.33333333],
          [0.,   0.,   1.33333333]],
         [[1.33333333,  -6.33333333,   3.66666667],
          [0.,   8.33333333, -10.33333333],
          [0.,   0.,   3.33333333]]]
    # Check for the correct answer on a uniform grid.
    beta = jiang_shu_smoothness_coefficients(3, [1, 2, 3, 4, 5, 6, 7])
    assert(np.testing.assert_almost_equal(beta[3], correct))

    # Make sure it doesn't fail in the case of a non-uniform grid.
    # How to check correctness?
    beta2 = jiang_shu_smoothness_coefficients(3, [1, 2, 3, 4, 5, 6, 7])
    assert(beta2 is not None)
