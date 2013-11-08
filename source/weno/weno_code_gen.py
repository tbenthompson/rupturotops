"""
PyWENO non-uniform reconstruction routines.

Original code written by Matthew Emmett as part of PyWENO.
Edited by Ben Thompson.
"""


import numpy as np
import sympy
# from sympy.utilities.codegen import codegen

from core.debug import _DEBUG

#
def module_name(k, type, f=None):
    """
    Returns the proper module name for WENO
    coefficients.

    The modules are created as [f]_[type]_[k]

    :param type: "smoothness", "weights", or "coefficients"
    :param k: reconstruction order
    :param f: base prefix for filenames
    """
    if f is None:
        f = 'weno'
    name = f + '_' + type + '_' + str(k)
    return name

def test_module_name():
    assert(module_name(3, 'abcdef') == 'weno_abcdef_3')

def polynomial_interpolator(x, y):
    """Build a symbolic polynomial that interpolates the points (x_i, y_i).

    The returned polynomial is a function of the SymPy variable x.
    """
    k, xi = len(x), sympy.var('x')
    sum_i = 0
    for i in range(k):
        ns = range(k)
        ns.remove(i)

        num, den = 1, 1
        for n in ns:
            num *= xi - x[n]
            den *= x[i] - x[n]

        sum_i += num / den * y[i]

    return sum_i

def test_polynomial_interpolator():
    x = [1.0, 2.0, 3.0]
    y = [2.0, 1.0, 2.0]
    p = polynomial_interpolator(x, y)
    x = sympy.var('x')
    assert(p.subs(x, 1.0) == 2.0)
    assert(p.subs(x, 2.0) == 1.0)
    assert(p.subs(x, 3.0) == 2.0)
    #polynomial should be a quadratic (x-1)**2
    assert(p.subs(x, 4.0) == 5.0)


def primitive_polynomial_interpolator(x, y):
    """Build a symbolic polynomial that approximates the primitive
    function f such that f(x_i) = sum_j y_j * (x_{j+1} - x_{j}).

    Note: The x argument should be list that is one element longer
    than the y list.

    The returned polynomial is a function of the SymPy variable 'x'.
    """

    Y = [0]
    for i in range(len(y)):
        Y.append(Y[-1] + (x[i + 1] - x[i]) * y[i])

    return polynomial_interpolator(x, Y)


def test_prim_polynomial_interpolator():
    x = [1.0, 2.0, 3.0]
    y = [2.0, 1.0]
    p = primitive_polynomial_interpolator(x, y)
    x = sympy.var('x')
    assert(p.subs(x, 1.0) == 0.0)
    assert(p.subs(x, 2.0) == 2.0)
    assert(p.subs(x, 3.0) == 3.0)
    assert(p.subs(x, 4.0) == 3.0)

def _pt(a, b, x):
    """Map x in [-1, 1] to x in [a, b]."""
    half = sympy.sympify(1/2)
    w = half * (b - a)
    c = half * (a + b)

    return w * x + c

def test_pt():
    assert(_pt(-2.0, 0.0, 0.5) == -0.5)
    assert(_pt(0.0, 3.0, 0.3) == 1.95)
    assert(_pt(-3.0, 1.0, 0.0) == -1.0)
    assert(_pt(-3.0, 1.0, 0.0) == -1.0)
    x = sympy.var('x')
    a = sympy.var('a')
    assert(_pt(a, 0, x) == 0.5 * (-a * x + a))


def coeffs_fnc_name(k, d, r, j):
    return 'reconstruction_coeffs_' + str(k) + '_' + str(d) + '_' + str(j)

def coeffs(k, d):
    # build arrays of cell boundaries and cell averages
    xs = [sympy.var('x%d' % j) for j in range(k + 1)]
    fs = [sympy.var('f%d' % j) for j in range(k)]

    x = sympy.var('x')
    z = _pt(xs[k - 1], xs[k], x)

    # compute reconstruction coefficients for each left shift r
    fnc_list = []
    poly = primitive_polynomial_interpolator(xs, fs).diff(x, d + 1)
    _DEBUG(k)
    poly = poly.simplify().expand()
    for j in range(k):
        args = xs
        args.extend(fs)
        args.append(z)
        fnc = poly.coeff(fs[j]).simplify()
        fnc_name = coeffs_fnc_name('reconstruction_coeffs', k, d, j)
        fnc_list.append((fnc, fnc_name, args))
    return fnc_list

def test_reconstruction_coeffs_gen():
    result = coeffs(3, 0)
    pass #how to test code generation? probably just wait until we get the results


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

    p = primitive_polynomial_interpolator(xs, ys)
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
                # l = autowrap([coeffs_expr], args=args,
                # tempdir=autogen_fortran/
                inner_coeffs_list.append(l)
            coeffs_list.append(inner_coeffs_list)
        fncs.append(coeffs_list)
    _DEBUG(2)


# def generate_function(coeffs_expr, args, directory, name):

# This test takes forever to run, so it's okay to turn it off unless it's
# needed.


def test_jiang_shu_smoothness_create():
    jiang_shu_smoothness_create(3)


