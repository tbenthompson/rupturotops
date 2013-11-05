import numpy as np


class Mesh(object):

    """
    Mesh design and storage. Currently the left edge is fixed at 0.0
    """

    def __init__(self, delta_x=(0.1 * np.ones(10))):
        """
        Sum the spacings to get cell centers,
        but the cell center is at the halfway point, so we have to
        subtract half of delta_x again.
        """
        self.delta_x = delta_x
        right_edges = np.cumsum(self.delta_x)
        self.centers = right_edges - self.delta_x / 2.0

        self.edges = np.insert(right_edges, 0, 0.0)

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
        return self.edges[-1]

    def extend_edges(self, extension):
        delta_xi = self.delta_x[0]
        delta_xf = self.delta_x[-1]
        padded_edges = self.edges
        for i in range(1, extension + 1):
            padded_edges = np.insert(padded_edges, 0,
                                     self.edges[0] - i * delta_xi)
            padded_edges = np.append(padded_edges,
                                     self.edges[-1] + i * delta_xf)
        return padded_edges

def test_extend_edges():
    delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    m = Mesh(delta_x)
    extended = m.extend_edges(2)
    assert(extended[0] == extended[2] - 1.0)
    assert(extended[1] == extended[2] - 0.5)
    assert(extended[-2] == extended[-3] + 2.0)
    assert(extended[-1] == extended[-3] + 4.0)


def test_mesh_simple():
    delta_x = np.array([0.5, 1.0, 0.1, 2.0])
    correct_cen = np.array([0.25, 1.0, 1.55, 2.6])
    correct_edge = np.array([0.0, 0.5, 1.5, 1.6, 3.6])
    m = Mesh(delta_x)
    assert((m.x == correct_cen).all())
    assert((m.edges == correct_edge).all())
    assert(m.domain_width == 3.6)
    assert(m.right_edge == 3.6)
    assert(m.left_edge == 0.0)
