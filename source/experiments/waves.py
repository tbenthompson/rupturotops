import dolfin as dfn
import numpy as np
# from math import exp
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG
# import ufl.operators
from core.experiment import Experiment
from dolfin import MPI


class LinearProblem:
    def __init__(self, a, l, bcs, solver):
        self.a = a
        self.l = l
        self.bcs = bcs
        self.solver = solver

    def solve(self,X):
        A = dfn.assemble(self.a)
        b = dfn.assemble(self.l)
        for i in self.bcs:
            i.apply(A)
            i.apply(b)
        self.solver.solve(A, X, b)

class Waves(Experiment):
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
            self.mesh = dfn.RectangleMesh(self.params.x_min, self.params.x_min, self.params.x_max,
                                          self.params.x_max, self.params.x_points, self.params.x_points)
        self.VS = dfn.FunctionSpace(self.mesh, 'CG', 1) #velocity fnc space
        self.SS = dfn.VectorFunctionSpace(self.mesh, 'CG', 1) #stress fnc space
        self.MS = self.VS * self.SS

        class InitCond(dfn.Expression):
            def eval(self, value, x):
                value[0] = np.exp(- (x[0] * x[0]) / 1.0)
                # value[0] = np.exp(- (x[0] * x[0] + x[1] * x[1]) / 50000000.0)
                value[1] = 0.0
                # value[2] = 0.0

            def value_shape(self):
                return (2,)

        # initial_both = dfn.Expression(code2D)
        # self.bc = dfn.DirichletBC(self.MS, dfn.Constant(), lambda x, on_bndry: on_bndry)

        vt, st = dfn.TestFunctions(self.MS)

        self.u = dfn.TrialFunction(self.MS)
        self.u0 = dfn.Function(self.MS)
        self.v = self.u[0]
        # self.s = dfn.as_vector([self.u[1], self.u[2]]) # current soln
        self.s = dfn.as_vector([self.u[1]]) # current soln
        self.v0, self.s0 = dfn.split(self.u0)  # previous soln

        init = InitCond()
        self.u0.interpolate(init)

        #new solution
        theta = 0.5
        midpoint_s = (1.0 - theta) * self.s + theta * self.s0
        # midpoint_s = theta * self.s0
        midpoint_v = (1.0 - theta) * self.v + theta * self.v0
        # midpoint_v = theta * self.v0

        self.dt_factor = dfn.Constant(1 / self.params.delta_t)

        self.form = dfn.Constant(np.sqrt(self.material.shear_modulus / self.material.density)) * \
            dfn.div(midpoint_s) * vt * dfn.dx - \
            dfn.Constant(np.sqrt(self.material.shear_modulus / self.material.density)) *\
            midpoint_v * dfn.div(st) * dfn.dx + \
            self.dt_factor * dfn.dot(self.s, st) * dfn.dx - \
            self.dt_factor * dfn.dot(self.s0, st) * dfn.dx + \
            self.dt_factor * self.v * vt * dfn.dx - \
            self.dt_factor * self.v0 * vt * dfn.dx

        self.solver = LinearProblem(dfn.lhs(self.form), dfn.rhs(self.form), [], dfn.LUSolver())
        # J =
        # self.problem = dfn.NonlinearVariationalProblem(self.form + self.l, self.u, [self.bc],
            # dfn.derivative(self.form, self.u, self.du))
        # self.solver = NonlinearVariationalSolver(self.problem)
        # prm = self.solver.parameters
        # prm['newton_solver']['absolute_tolerance'] = 1E-8
        # prm['newton_solver']['relative_tolerance'] = 1E-7
        # prm['newton_solver']['maximum_iterations'] = 25
        # prm['newton_solver']['relaxation_parameter'] = 1.0
        # prm['linear_solver'] = 'lu'
        # prm['preconditioner'] = 'jacobi'
        # prm['krylov_solver']['absolute_tolerance'] = 1E-9
        # prm['krylov_solver']['relative_tolerance'] = 1E-7
        # prm['krylov_solver']['maximum_iterations'] = 1000
        # prm['krylov_solver']['gmres']['restart'] = 40
        # prm['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

    def _compute(self):
        # Supress some FEniCS output
        dfn.set_log_level(16)
        t = 0.0
        myplot = dfn.plot(self.v0)
        myplot2 = dfn.plot(self.s0)
        vtk_file = dfn.File("../data/waves_fem_output/elasticity.pvd")
        self.u = dfn.Function(self.MS)
        while t <= self.params.t_max:
            t += self.params.delta_t
            self.solver.solve(self.u.vector())
            self.u0.vector()[:] = self.u.vector()[:]
            myplot.plot(self.u.split()[0])
            myplot2.plot(self.u.split()[1])
            vtk_file << self.u.split()[0]
        # Solve the problem

    def _visualize(self):
        # pyp.plot(self.u_.vector().array())
        pass
