import numpy as np
class RiemannSolver(object):
    """
    Really it's a Godunov flux

    This interface could easily be used with a TVD flux too...
    """
    def __init__(self):
        pass

    def split_velocity(self, v):
        rightwards = np.where(v > 0, v, 0)
        leftwards = np.roll(-np.where(v < 0, v, 0), -1)
        return rightwards, leftwards

    def compute(self, recon_left, recon_right, v):
        rightwards_v, leftwards_v = self.split_velocity(v)
        left_edge_u = leftwards_v * recon_left
        right_edge_u = rightwards_v * recon_right

        # For solving a Riemann problem, the zeroth
        # order contribution to the flux should be the difference
        # in the left and right values at a boundary
        # outflow comes from the right side and inflow from the left
        # accounting for the sign
        left_side_of_bnd = np.roll(right_edge_u, 1)
        right_side_of_bnd = left_edge_u
        #this the rightwards flux on the left boundary of a given cell
        total_rightwards_flux = left_side_of_bnd - right_side_of_bnd
        return total_rightwards_flux

#--------------------------------------------------------------------------
# TESTS
#--------------------------------------------------------------------------
def test_riemann_solver_compute():
    v = np.array([1, 1, 1, 1, 1])
    recon_left = np.array([0, 0, 1, 1, 0])
    recon_right = np.array([0, 0, 1, 1, 0])
    correct = np.array([0, 0, 0, 1, 1])
    rs = RiemannSolver()
    result = rs.compute(recon_left, recon_right, v)
    np.testing.assert_almost_equal(result, correct)


def test_split_vel():
    v = np.array([1, -1, 1])
    rs = RiemannSolver()
    right, left = rs.split_velocity(v)
    assert((left == [1, 0, 0]).all())
    assert((right == [1, 0, 1]).all())
