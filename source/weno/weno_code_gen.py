"""
PyWENO non-uniform reconstruction routines.

Original code written by Matthew Emmett as part of PyWENO.
Edited by Ben Thompson.
"""

import subprocess
import numpy as np
import numpy.f2py as f2py
import sympy
from sympy.utilities.codegen import codegen

from core.debug import _DEBUG

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
    half = sympy.sympify(1.0/2.0)
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


def coeffs_fnc_name(k, d, j):
    return 'reconstruction_coeffs_' + str(k) + '_' + str(d) + '_' + str(j)

def coeffs(k, d):
    # build arrays of cell boundaries and cell averages
    xs = [sympy.var('x%d' % j) for j in range(k + 1)]
    fs = [sympy.var('f%d' % j) for j in range(k)]

    x = sympy.var('x')
    # z = _pt(xs[k - 1], xs[k], x)

    # compute reconstruction coefficients for each left shift r
    fnc_list = []
    poly = primitive_polynomial_interpolator(xs, fs).diff(x, d + 1)
    for j in range(k):
        args = xs[:]
        args.append(x)
        fnc = sympy.Poly(poly, fs[j]).coeff_monomial(fs[j]).simplify()
        fnc_name = coeffs_fnc_name(k, d, j)
        fnc_list.append((fnc, fnc_name, args))
    return fnc_list

def test_reconstruction_coeffs_gen():
    result = coeffs(3, 0)
    assert(len(result) == 3)
    assert(result[0][0] != 0)
    assert(result[1][2].count(sympy.var('x1')) == 1)
    result2 = coeffs(3, 2)
    assert(result2)
    pass #how to test code generation? probably just wait until we get the results


def smoothness_fnc_name(k, r, i, j):
    """
    Arguments:
        :param k: The order of the polynomial reconstruction.
        :param i: The first dimension of the smoothness kernel position.
        :param j: The second dimension of the smoothness kernel position.
    """
    return 'smoothness_' + str(k) + '_' + str(r) + '_' + str(i) + '_' + str(j)


def smoothness(k):
    """
    Compute the Jiang-Shu smoothness coefficient functions and output them in a form ready
    to be compiled to fortran.
    """

    # the integration variable (represents position within the cell)
    x = sympy.var('x')

    # build array of cell boundaries and cell averages (sympy vars x[i])
    xs = []
    for j in range(k + 1):
        xs.append(sympy.var('x%d' % j))
    # and build the array of cell averages
    ys = []
    for j in range(k):
        ys.append(sympy.var('y%d' % j))

    # The upper and lower boundaries of the cell, used as the integration
    # bounds.
    b1 = sympy.var('b1')
    b2 = sympy.var('b2')

    p = primitive_polynomial_interpolator(xs, ys)
    p = p.as_poly(x)
    p = p.diff(x)

    ppp = []
    for r in range(0, k):
        # sum of L^2 norms of derivatives
        s = 0
        for j in range(1, k):
            pp = (p.diff((0, j))) ** 2
            pp = pp.integrate(x)
            pp = (b2 - b1) ** (2 * j - 1) * (
                pp.subs(x, b2) - pp.subs(x, b1))
            s = s + pp
        ppp.append(s.simplify().expand())

    # Return functions and arguments in a form ready to be compiled to fortran
    # expressions that can be called from the live code.
    fncs = []
    for r in range(0, k):
        for i in range(0, k):
            for j in range(0, k):
                args = [xs[m] for m in range(0, k + 1)]
                # I suspect that I could carefully redesign this so that
                # b1 and b2 would be drawn from the xs array
                args.append(b1)
                args.append(b2)
                fnc = ppp[r].coeff(ys[i] * ys[j]).simplify()
                fnc_name = smoothness_fnc_name(k, r, i, j)
                fncs.append((fnc, fnc_name, args))
    return fncs


def test_jiang_shu_smoothness_create():
    assert(smoothness(3))


def create_fncs(k):
    """
    This function uses the sympy autowrap module to create fortran functions
    with f2py. It is massively faster at runtime than directly calling the
    sympy functions using subs. As a tradeoff, this takes a couple minutes
    to generate the fortran functions.
    """
    filename_prefix = 'weno_' + str(k)
    fncs = []
    for d in range(0, 1):
        fncs.extend(coeffs(k, d))
    fncs.extend(smoothness(k))
    args_list = [f[2] for f in fncs]
    fncs_and_names = [(f[1], f[0]) for f in fncs]
    source_file = filename_prefix + '.f90'
    with open(source_file, 'w') as ff:
        ff.write('! -*- f90 -*-\n')
        for i in range(len(fncs_and_names)):
            code = codegen(fncs_and_names[i], "F95", "autoweno",
                            argument_sequence=args_list[i])
            ff.write(code[0][1])
    subprocess.call('f2py -c ' + source_file + ' -m ' + filename_prefix,
                    shell=True)
    subprocess.call('rm ' + source_file, shell=True)
    module = __import__(filename_prefix)
    assert(module)


def test_create_fncs():
    create_fncs(3)
