import numpy as np

class GodunovDeriv(object):
    def __init__(self, reconstructor, riemann, v):
        self.v = v
        self.reconstructor = reconstructor
        self.riemann = riemann

    def compute(self, t, now, d=0):
        recon_left = lambda data: self.reconstructor.compute(data,
                                                             'left', d)
        recon_right = lambda data: self.reconstructor.compute(data,
                                                              'right', d)
        #second order flux limiting could be implemented here!
        rightwards_flux = self.riemann.compute(recon_left,
                                               recon_right,
                                               now,
                                               self.v)
        return rightwards_flux

class SimpleFlux(object):
    def __init__(self, mesh, bc, deriv):
        self.mesh = mesh
        self.bc = bc
        self.deriv = deriv
        self.delta_x = self.bc.extend_dx(self.mesh.delta_x)

    def compute(self, t, now, get_derivs=False):
        """
        Returns a list of terms, one for each derivative in the taylor series.
        """
        now = self.bc.compute(t, now)
        total_flux = []
        for i in range(3):
            if i > 0 and not get_derivs:
                break
            rightwards_flux = self.deriv.compute(t, now, d=i)
            # The total flux should be the flux coming in from the right
            # minus the flux going out the left.
            leftwards_flux = -np.roll(rightwards_flux, -1)

            tf = (rightwards_flux + leftwards_flux) / self.delta_x
            total_flux.append(tf)
        if get_derivs:
            return total_flux
        return total_flux[0]

#--------------------------------------------------------------------------
# TESTS
#--------------------------------------------------------------------------

from experiments.wave_forms import square
from core.mesh import Mesh
from experiments.weno import WENO
from experiments.boundary_conds import PeriodicBC
from experiments.riemann_solver import RiemannSolver

def test_spatial_deriv():
    m = Mesh(0.1 * np.ones(50))
    d = GodunovDeriv(WENO(m), RiemannSolver(), np.ones(54))
    sd = SimpleFlux(m, PeriodicBC(), d)
    derivative = sd.compute(0, np.pad(square(m.x), 2, 'constant'))
    assert(np.sum(derivative) <= 0.005)
    for i in range(0, len(derivative)):
        if i == 7:
            np.testing.assert_almost_equal(derivative[7], -10.0)
            continue
        if i == 12:
            np.testing.assert_almost_equal(derivative[12], 10.0)
            continue
        np.testing.assert_almost_equal(derivative[i], 0.0)
