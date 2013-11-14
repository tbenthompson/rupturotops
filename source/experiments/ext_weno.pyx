import numpy as np
import cython


def mult_with_smoothness(int cells, int half_width, double[:] padded,
                         double[:, :, :, :] smoothness):
    cdef double[:, :] beta = np.zeros((cells, half_width),
                                      dtype=np.float64)
    cdef int l, r, i, j
    for l in range(cells):
        for r in range(half_width):
            for i in range(0, half_width):
                for j in range(0, half_width):
                    beta[l, half_width - 1 - r] += smoothness[l,r,i,j] * \
                        padded[half_width - 1 + l - r + i] * \
                        padded[half_width - 1 + l - r + j]
    # note that this result is in the backwards pyWENO form
    # retval = [beta3, beta2, beta1] in the Shu 2009 notation
    return beta


def mult_polynomials(int cells, int half_width, double[:, :] weights,
                     double[:, :] small_polynomials):
    cdef double[:] reconstruction = np.zeros((cells), dtype=np.float64)
    cdef int i, j
    for i in range(cells):
        for j in range(half_width):
            reconstruction[i] += weights[i, j] * small_polynomials[i, j]
    return reconstruction


def scaled_weights(int cells, int half_width, double eps,
                   double[:, :] weights,
                   double[:, :] beta):
    cdef double[:, :] scaled = np.zeros((cells, half_width),
                                        dtype=np.float64)
    cdef double sum
    cdef double denom_pre
    cdef int i, j
    for i in range(cells):
        sum = 0.0
        for j in range(half_width):
            denom_pre = (eps + beta[i, half_width - 1 - j])
            scaled[i, half_width - 1 - j] = weights[i, j] / \
                (denom_pre * denom_pre)
            sum += scaled[i, half_width - 1 - j]
        if sum == 0.0:
            continue
        for j in range(half_width):
            scaled[i, j] /= sum
    return scaled


def mult_with_coeffs(int cells, int half_width, double[:] padded,
                     double[:, :, :] coeffs):
    cdef double[:, :] retval = np.zeros((cells, half_width),
                                        dtype=np.float64)
    cdef int i, r, j
    cdef int center
    for i in range(cells):
        center = half_width - 1 + i
        for r in range(half_width):
            # pyWENO does a weird flipping with its smoothness coefficients
            # so we go backwards
            for j in range(half_width):
                # note that this result is in the backwards pyWENO form
                # retval = [poly3, poly2, poly1] in the Shu 2009 notation
                retval[i, half_width - 1 - r] += coeffs[i, r, j] *\
                    padded[center - r + j]
    return retval
