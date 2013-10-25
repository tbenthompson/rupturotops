import dolfin as dfn
import numpy as np
# from math import exp
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG
# import ufl.operators
from core.experiment import Experiment
from core.constants import consts


class ShearHeatingFenics(Experiment):
    """
    Solving the shear heating equations with FEnICS
    """
    def _initialize(self):
        # This nondimensionalization code is replicated from shear_heating.py and should
        # eventually be migrated to a central location

        self.data.length_scale = np.sqrt(self.material.density /
                                         self.material.shear_modulus) * \
            self.material.thermal_diffusivity
        self.data.time_scale = self.material.density * self.material.thermal_diffusivity / \
            self.material.shear_modulus
        self.data.stress_scale = self.material.shear_modulus
        self.data.inv_prandtl = self.material.creep_constant * self.material.density * \
            self.material.thermal_diffusivity * \
            self.material.shear_modulus ** (self.material.stress_exponent - 1)
        self.data.eckert = self.material.shear_modulus / (self.params.delta_temp *
                                                          self.material.specific_heat *
                                                          self.material.density)

        self.data.x_min = self.params.x_min / self.data.length_scale
        self.data.x_max = self.params.x_max / self.data.length_scale
        self.data.t_max = self.params.t_max / self.data.time_scale
        self.data.delta_t = self.params.delta_t / self.data.time_scale
        self.data.initial_temp_start_time = self.params.initial_temp_start_time / self.data.time_scale
        self.data.stress = self.params.stress / self.data.stress_scale
        self.data.source_term = self.params.source_term * self.data.time_scale / self.params.delta_temp

        # Create mesh and define function space
        self.mesh = dfn.IntervalMesh(self.params.x_points, self.data.x_min, self.data.x_max)
        self.V = dfn.FunctionSpace(self.mesh, 'CG', 2)

        # Define boundary conditions -- gaussian initial conditions
        self.u0 = dfn.Constant(0.0)
        self.bc = dfn.DirichletBC(self.V, dfn.Constant(0.0), lambda x, on_bndry: on_bndry)

        class SourceTermExpr(dfn.Expression):
            def __init__(self, src):
                self.src = src

            def eval(self, value, x):
                value[0] = 0.0
                if x[0] == 0.0:
                    value[0] = self.src
        source = SourceTermExpr(self.data.source_term)

        #initial conditions
        self.u_ = dfn.interpolate(self.u0, self.V)
        self.u_old = dfn.interpolate(self.u0, self.V)
        self.s_ = dfn.interpolate(dfn.Constant(0.0), self.V)
        self.s_old = dfn.interpolate(dfn.Constant(0.0), self.V)

        # Define variational problem
        self.du = dfn.TrialFunction(self.V)
        self.du2 = dfn.TrialFunction(self.V)
        self.v = dfn.TestFunction(self.V)
        self.T = self.data.delta_t * \
            dfn.inner(dfn.nabla_grad(self.u_ + self.s_), dfn.nabla_grad(self.v)) * \
            dfn.dx + \
            self.u_ * self.v * dfn.dx - \
            self.u_old * self.v * dfn.dx - \
            self.data.delta_t * source * self.v * dfn.dx

        self.SH = self.s_ * self.v * dfn.dx - \
            self.s_old * self.v * dfn.dx - \
            self.data.delta_t * self.data.inv_prandtl * self.data.eckert * \
            self.data.stress ** (self.material.stress_exponent + 1) * \
            dfn.exp(-(self.material.activation_energy / consts.R) /
                    (self.params.low_temp + self.params.delta_temp * (self.u_ + self.s_))) * \
            self.v * dfn.dx

        self.V = self.v_ * self.v * dfn.dx - \


        # J must be a Jacobian (Gateaux derivative in direction of du)
        self.T_J = dfn.derivative(self.T, self.u_, self.du)
        self.SH_J = dfn.derivative(self.SH, self.s_, self.du2)
        #info(prm, True)

    def _compute(self):
        dfn.set_log_level(16)

        problem = dfn.NonlinearVariationalProblem(self.F, self.u_, self.bc, self.J)
        problem2 = dfn.NonlinearVariationalProblem(self.F2, self.s_, self.bc, self.J2)
        solver = dfn.NonlinearVariationalSolver(problem)
        solver2 = dfn.NonlinearVariationalSolver(problem2)
        _DEBUG()
        # t = self.data.initial_temp_start_time + self.data.delta_t
        t = self.data.delta_t
        t_max = self.data.t_max
        while t <= t_max:
            solver.solve()
            solver2.solve()
            t += self.data.delta_t
            self.u_old.assign(self.u_)
            self.s_old.assign(self.s_)
            # percentage = abs(((t / t_max) * 1000) - np.floor((t / t_max) * 1000))
            # if percentage <= ((1000 * self.data.delta_t) / t_max):
            pyp.plot(self.s_.vector().array())
        pyp.show()

    def _visualize(self):
        pyp.plot(self.u_.vector().array())
        pyp.show()
