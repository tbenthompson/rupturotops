import dolfin as dfn
import numpy as np
# from math import exp
# from matplotlib import pyplot as pyp
# from pdb import set_trace as _DEBUG
# import ufl.operators
from core.experiment import Experiment
# from dolfin import MPI


class LinearProblem:
    def __init__(self, a, l, bcs, solver):
        self.a = a
        self.l = l
        self.bcs = bcs
        self.solver = solver

    def solve(self, X):
        A = dfn.assemble(self.a)
        b = dfn.assemble(self.l)
        for i in self.bcs:
            i.apply(A)
            i.apply(b)
        self.solver.solve(A, X, b)


class DGWaves(Experiment):
    """
    Solving the wave equation with FEnICS

    The useful references are the auto-adaptive NS solver,
    the Cahn-Hilliard equations, and the nonlinear/time
    dependent tutorials online
    """
    def _initialize(self):
        dfn.parameters["form_compiler"]["cpp_optimize"] = True
        dfn.parameters["form_compiler"]["optimize"] = True
        d = 1
        # Create mesh and define function space

        if d is 1:
            self.mesh = dfn.IntervalMesh(self.params.x_points, self.params.x_min, self.params.x_max)
        elif d is 2:
            self.mesh = dfn.RectangleMesh(self.params.x_min, self.params.y_min, self.params.x_max,
                                          self.params.y_max, self.params.x_points, self.params.y_points)
        self.FS = dfn.FunctionSpace(self.mesh, 'DG', 1)  # velocity fnc space
        self.SS = dfn.VectorFunctionSpace(self.mesh, 'DG', 1)
        self.MS = self.FS * self.SS

        class InitCond(dfn.Expression):
            def eval(self, value, x):
                value[0] = np.exp(- (x[0] * x[0]) / 1.0)
                # value[0] = np.exp(- (x[0] * x[0] + x[1] * x[1]) / 1.0)
                value[1] = 0.0
                # value[2] = 0.0

            def value_shape(self):
                return (2,)

        # initial_both = dfn.Expression(code2D)
        # self.bc = dfn.DirichletBC(self.MS, dfn.Constant((0.0, 0.0)), lambda x, on_bndry: on_bndry)

        self.vt, self.st = dfn.TestFunctions(self.MS)

        self.u = dfn.TrialFunction(self.MS)
        self.u0 = dfn.Function(self.MS)
        self.v = self.u[0]
        self.s = dfn.as_vector([self.u[1]])
        # self.s = dfn.as_vector([self.u[1], self.u[2]])
        self.v0, self.s0 = dfn.split(self.u0)

        self.init = InitCond()
        self.u0.interpolate(self.init)

        self.dt_factor = dfn.Constant(1 / self.params.delta_t)
#
        c = dfn.Constant(np.sqrt(self.material.shear_modulus / self.material.density))

        n = dfn.FacetNormal(self.mesh)
        alpha = 0.5
        # upwindS = c * self.s0
        # upwindS = dfn.dot(dfn.avg(upwindS) + alpha * dfn.jump(upwindS), n)
        # upwindV = c * self.v0 * n
        # upwindV = dfn.avg(c * self.v0) + alpha * dfn.jump(upwindV)

        self.form = -c * (dfn.dot(self.s0, dfn.grad(self.vt)) * dfn.dx +
                          self.v0 * dfn.div(self.st) * dfn.dx)
        self.form += dfn.jump(self.vt) * \
            dfn.dot(n('+'), dfn.avg(c * self.s0) + 0.5 * dfn.jump(self.s0)) * \
            dfn.dS
        self.form += dfn.dot(dfn.jump(self.st),
                             n('+') * (dfn.avg(c * self.v0) + 0.5 * dfn.jump(self.v0))) * \
            dfn.dS
        self.form += -self.dt_factor * (dfn.dot(self.s, self.st) * dfn.dx -
                                        dfn.dot(self.s0, self.st) * dfn.dx +
                                        self.v * self.vt * dfn.dx -
                                        self.v0 * self.vt * dfn.dx)

        self.solver = LinearProblem(dfn.lhs(self.form), dfn.rhs(self.form), [], dfn.LUSolver())

    def _compute(self):
        # Supress some FEniCS output
        dfn.set_log_level(16)
        t = 0.0
        myplot = dfn.plot(self.v0)
        myplot2 = dfn.plot(self.s0)
        self.u = dfn.Function(self.MS)
        while t <= self.params.t_max:
            t += self.params.delta_t
            self.solver.solve(self.u.vector())
            self.u0.vector()[:] = self.u.vector()[:]
            myplot.plot(self.u.split()[0])
            myplot2.plot(self.u.split()[1])
        # Solve the problem

    def _visualize(self):
        # pyp.plot(self.u_.vector().array())
        pass
