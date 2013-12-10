import numpy as np
from core.debug import _DEBUG


class SimpleFlux(object):

    def __init__(self, reconstructor, v, mesh, bc, assume_constant_v=True):
        self.mesh = mesh
        self.v = v
        self.reconstructor = reconstructor
        self.bc = bc
        self.delta_x = self.mesh.delta_x
        self.assume_constant_v = assume_constant_v
        # delta_x should be the same size as "now" so we can use it to
        # instantiate the array
        self.padded = np.pad(self.mesh.delta_x.copy(), 2, 'constant')

    def split_velocity(self, v):
        if self.assume_constant_v:
            if v[0] > 0:
                rightwards = v
                leftwards = np.zeros_like(v)
            if v[0] < 0:
                leftwards = v
                rightwards = np.zeros_like(v)
        else:
            rightwards = np.where(v > 0, v, 0)
            leftwards = np.roll(-np.where(v < 0, v, 0), -1)
        return rightwards, leftwards

    def solve_riemann(self, recon_left, recon_right, now, v):
        rightwards_v, leftwards_v = self.split_velocity(v)
        if not np.all(leftwards_v == 0.0):
            left_edge_u = leftwards_v * recon_left(now)
        else:
            left_edge_u = np.zeros(len(now))
        if not np.all(rightwards_v == 0.0):
            right_edge_u = rightwards_v * recon_right(now)
        else:
            right_edge_u = np.zeros(len(now))

        # For solving a Riemann problem, the zeroth
        # order contribution to the flux should be the difference
        # in the left and right values at a boundary
        # outflow comes from the right side and inflow from the left
        # accounting for the sign
        left_side_of_bnd = np.roll(right_edge_u, 1)
        right_side_of_bnd = left_edge_u
        # this the rightwards flux on the left boundary of a given cell
        total_rightwards_flux = left_side_of_bnd - right_side_of_bnd
        return total_rightwards_flux


    def godunov(self, t, now, d=0):
        recon_left = lambda data: self.reconstructor.compute(data,
                                                             'left', d)
        recon_right = lambda data: self.reconstructor.compute(data,
                                                              'right', d)
        # second order flux limiting could be implemented here!
        rightwards_flux = self.solve_riemann(recon_left,
                                               recon_right,
                                               now,
                                               self.v)
        return rightwards_flux

    def compute(self, t, now, get_derivs=False):
        """
        Returns a list of terms, one for each derivative in the taylor series.
        """
        self.padded[2:-2] = now
        self.padded = self.bc.compute(t, self.padded)
        total_flux = []
        for i in range(3):
            if i > 0 and not get_derivs:
                break
            rightwards_flux = self.godunov(t, self.padded, d=i)
            # The total flux should be the flux coming in from the right
            # minus the flux going out the left.
            leftwards_flux = -np.roll(rightwards_flux, -1)

            tf = (rightwards_flux + leftwards_flux)[2:-2] / self.delta_x
            total_flux.append(tf)
        if get_derivs:
            return total_flux
        return total_flux[0]

#--------------------------------------------------------------------------
# TESTS
#--------------------------------------------------------------------------

from rupturotops.wave_forms import square
from core.mesh import Mesh
from rupturotops.weno import WENO
from rupturotops.boundary_conds import PeriodicBC

def test_riemann_solver_compute():
    v = np.array([1, 1, 1, 1, 1])
    recon_left = lambda x: np.array([0, 0, 1, 1, 0])
    recon_right = lambda x: np.array([0, 0, 1, 1, 0])
    correct = np.array([0, 0, 0, 1, 1])
    m = Mesh(0.1 * np.ones(50))
    sd = SimpleFlux(WENO(m), np.ones(50), m, PeriodicBC())
    result = sd.solve_riemann(recon_left, recon_right, [0,0,0,0,0], v)
    np.testing.assert_almost_equal(result, correct)


def test_split_vel():
    v = np.array([1, -1, 1])
    m = Mesh(0.1 * np.ones(50))
    sd = SimpleFlux(WENO(m), np.ones(50), m, PeriodicBC(), assume_constant_v=False)
    right, left = sd.split_velocity(v)
    assert((left == [1, 0, 0]).all())
    assert((right == [1, 0, 1]).all())

def test_spatial_deriv():
    m = Mesh(0.1 * np.ones(50))
    sd = SimpleFlux(WENO(m), np.ones(54), m, PeriodicBC())
    derivative = sd.compute(0, square(m.x))
    assert(np.sum(derivative) <= 0.005)
    for i in range(0, len(derivative)):
        if i == 5:
            np.testing.assert_almost_equal(derivative[i], -10.0)
            continue
        if i == 10:
            np.testing.assert_almost_equal(derivative[i], 10.0)
            continue
        np.testing.assert_almost_equal(derivative[i], 0.0)
