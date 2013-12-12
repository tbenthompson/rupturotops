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
from linear_viscoelastic.constant_slip_maxwell import velocity_solution


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
            self.velocity = self.vel_solver.update(t, dt, Szx, Szy)
            self.strs_solver.helmholtz_projection(t, dt, self.velocity)
            t += dt
            pyp.figure(5)
            pyp.plot(self.vel_solver.velX_, -self.velocity[0, :])

    def _visualize(self):
        #velocity solution is wrong except in the surface elastic layer!!
        v_3 = velocity_solution(self.vel_solver.velX,
                                self.vel_solver.velY,
                                self.params.t_max,
                                self.params.t_r,
                                self.params.fault_depth,
                                self.params.elastic_depth,
                                self.params.fault_slip,
                                100)
        pyp.figure(1)
        pyp.xlabel('x (meters)')
        pyp.ylabel(r'$\sigma_{zx}$ (Pascals)')
        pyp.figure(2)
        pyp.xlabel('x (meters)')
        pyp.ylabel(r'$\sigma_{zy}$ (Pascals)')
        pyp.figure(3)
        pyp.plot(self.vel_solver.velX_, v_3[0, :], label='analytical')
        pyp.plot(self.vel_solver.velX_, -self.velocity[0,:], label='projection')
        pyp.xlabel('x (meters)')
        pyp.ylabel('v (m/s)')
        pyp.legend()
        pyp.figure(4)
        im = pyp.imshow(-self.velocity, extent=[self.params.x_min, self.params.x_max,
                                                self.params.y_max, self.params.y_min])
        cb = pyp.colorbar()
        cb.set_label('v (m/s)')
        pyp.xlabel('x (meters)')
        pyp.ylabel('y (meters)')
        pyp.figure(5)
        pyp.xlabel('x (meters)')
        pyp.ylabel('v (m/s)')
        pyp.show()
        # pyp.figure(2)
        # im2 = pyp.imshow(v_3)
        # pyp.colorbar()
        # pyp.figure(3)
        # pyp.imshow(self.strs_solver.Szx)
        # pyp.figure(4)
        # pyp.imshow(self.strs_solver.Szy)
        # im.set_clim(im2.get_clim())
