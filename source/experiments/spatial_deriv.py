import numpy as np

class SpatialDeriv(object):
    def __init__(self, mesh, reconstructor, bc, riemann, v):
        self.mesh = mesh
        self.reconstructor = reconstructor
        self.bc = bc
        self.riemann = riemann
        self.v = v

    # @autojit()
    def compute(self, t, now):
        now = self.bc.compute(t, now)
        recon_left = self.reconstructor.compute(now, 'left')
        recon_right = self.reconstructor.compute(now, 'right')

        rightwards_flux = self.riemann.compute(recon_left,
                                               recon_right,
                                               self.v)
        # The total flux should be the flux coming in from the right
        # minus the flux going out the left.
        leftwards_flux = -np.roll(rightwards_flux, -1)
        total_flux = rightwards_flux + leftwards_flux
        return total_flux[2:-2] / self.mesh.delta_x

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
    sd = SpatialDeriv(m, WENO(m), PeriodicBC(), RiemannSolver(), np.ones(54))
    deriv = sd.compute(0, square(m.x))
    assert(np.sum(deriv) <= 0.005)
    for i in range(0, len(deriv)):
        if i == 5:
            np.testing.assert_almost_equal(deriv[5], -10.0)
            continue
        if i == 10:
            np.testing.assert_almost_equal(deriv[10], 10.0)
            continue
        np.testing.assert_almost_equal(deriv[i], 0.0)
