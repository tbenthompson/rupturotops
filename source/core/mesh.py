import numpy as np


class Mesh(object):

    """
    Mesh design and storage. Currently the left edge is fixed at 0.0
    """

    def __init__(self, delta_x=np.linspace(0, 1, 10)):
        """
        Sum the spacings to get cell centers,
        but the cell center is at the halfway point, so we have to
        subtract half of delta_x again.
        """
        self.delta_x = delta_x
        self.right_edges = np.cumsum(self.delta_x)
        self.centers = self.right_edges - self.delta_x / 2.0

    @property
    def x(self):
        return self.centers

    @property
    def left_edge(self):
        return 0.0

    @property
    def right_edge(self):
        return self.domain_width

    @property
    def domain_width(self):
        return self.right_edges[-1]


def test_mesh_simple():
    delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    correct_cen = np.array([0.25, 1.0, 1.55, 2.6])
    correct_edge = np.array([0.5, 1.5, 1.6, 3.6])
    m = Mesh(delta_x)
    assert((m.centers == correct_cen).all())
    assert((m.right_edges == correct_edge).all())
    assert(m.domain_width == 3.6)
    assert(m.right_edge == 3.6)
    assert(m.left_edge == 0.0)
