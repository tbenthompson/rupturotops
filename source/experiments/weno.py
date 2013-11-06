import scipy.io
import matplotlib.pyplot as pyp
from core.debug import _DEBUG
import numpy as np
# from numba import autojit

import pyweno.weno

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
