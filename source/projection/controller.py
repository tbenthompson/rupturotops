import time
import warnings
import dolfin as dfn
import scipy.integrate
import numpy as np
from dolfin import NonlinearVariationalProblem as NVP
from dolfin import NonlinearVariationalSolver as NVS
from matplotlib import pyplot as pyp
from core.experiment import Experiment
from core.constants import consts
from core.debug import _DEBUG
from projection.velocity import VelocitySolver
from projection.stress import StressSolver
# from projection.temperature import TempSolver




class ProjController(Experiment):

    """
    Solve a quasistatic viscoelastic problem using a projection scheme
    to maintain the divergence free stress condition.
    """

    def _initialize(self):
        # Create mesh and define function space
        # Subtract one from points to get the number of elements
        self.mesh = dfn.RectangleMesh(self.params.x_min, self.params.y_min,
                                      self.params.x_max, self.params.y_max,
                                      self.params.x_points - 1,
                                      self.params.y_points - 1)

        self.vel_solver = VelocitySolver(self.params, self.mesh)
        self.strs_solver = StressSolver(self.params)
        # TempSolver needs to be checked over
        # self.temp_solver = TempSolver(self.params, self.mesh)

    def _compute(self):
        dfn.set_log_level(16)
        dt = self.params.delta_t
        t = dt
        t_max = self.params.t_max
        while t <= t_max:
            Szx, Szy = self.strs_solver.update_momentum(t, dt)
            dvdx, dvdy, self.velocity = self.vel_solver.update(t, dt, Szx, Szy)
            self.strs_solver.helmholtz_projection(t, dt, dvdx, dvdy)
            t += dt
        pyp.show()

    def _visualize(self):
        from linear_viscoelastic.constant_slip_maxwell import velocity_solution
        t_r = self.params.viscosity / self.material.shear_modulus
        v_3 = velocity_solution(self.strs_solver.X_, self.params.t_max, t_r,
                       self.params.fault_depth, self.params.elastic_depth,
                       self.params.fault_slip)
        pyp.figure(1)
        pyp.plot(self.strs_solver.X_, v_3)
        pyp.figure(2)
        pyp.plot(self.strs_solver.X_, self.velocity[0,:])
        pyp.show()
        pyp.imshow(np.log(self.Szx))
        pyp.show()
