import numpy as np
from core.data_controller import DataController

class PeriodicBC(object):
    def __init__(self):
        pass

    def compute(self, t, now):
        ghosts = np.pad(now, 2, 'constant')
        ghosts[0] = ghosts[-4]
        ghosts[1] = ghosts[-3]  # np.sin(30 * t)
        ghosts[-2] = ghosts[2]
        ghosts[-1] = ghosts[3]
        return ghosts

def test_circular_boundary_conditions():
    pbc = PeriodicBC()
    x = np.linspace(0.0, 4.9, 50) + 0.05
    added_ghosts = pbc.compute(0, x)
    np.testing.assert_almost_equal(added_ghosts[0], 4.85)
    np.testing.assert_almost_equal(added_ghosts[1], 4.95)
    np.testing.assert_almost_equal(added_ghosts[-2], 0.05)
    np.testing.assert_almost_equal(added_ghosts[-1], 0.15)
