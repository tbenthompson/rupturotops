import dolfin as dfn
from dolfin import NonlinearVariationalProblem as NVP
from dolfin import NonlinearVariationalSolver as NVS
import numpy as np
# from math import exp
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG
# import ufl.operators
from core.experiment import Experiment
from core.constants import consts


class SourceTermExpr(dfn.Expression):

    def __init__(self, src):
        self.src = src

    def eval(self, value, x):
        value[0] = 0.0
        value[0] = self.src


class ShearHeatingFenics(Experiment):

    """
    Solving the shear heating equations with FEnICS
    """

    def _initialize(self):
        self.initial_T = dfn.Expression('A + B * exp(-x[0] * x[0] / 100)',
                                        A=self.params.background_temp,
                                        B=self.params.temp_pulse_size)
        # Create mesh and define function space
        self.mesh = dfn.RectangleMesh(self.params.x_min, self.params.y_min,
                                      self.params.x_max, self.params.y_max,
                                      self.params.x_points,
                                      self.params.y_points)
        self.V = dfn.FunctionSpace(self.mesh, 'CG', 2)

        self.bcT = dfn.DirichletBC(self.V,
                                   self.initial_T,
                                   lambda x, on_bndry: on_bndry)

        # source = SourceTermExpr(self.params.source_term)
        source = dfn.Expression('B * exp(-x[0] * x[0])',
                                B=self.params.source_term)

        # initial conditions
        self.T_ = dfn.interpolate(self.initial_T, self.V)
        self.T_old = dfn.interpolate(self.initial_T, self.V)

        # Define variational problem
        self.dT = dfn.TrialFunction(self.V)
        self.v = dfn.TestFunction(self.V)
        self.diffusive_term = self.params.delta_t * \
            self.material.thermal_diffusivity * \
            dfn.inner(dfn.nabla_grad(self.T_),
                      dfn.nabla_grad(self.v)) * dfn.dx + \
            self.T_ * self.v * dfn.dx - \
            self.T_old * self.v * dfn.dx

        self.source_term = -self.params.delta_t * \
            (1.0 / (self.material.specific_heat * self.material.density)) * \
            source * self.v * dfn.dx

        self.shear_heat_term = self.params.delta_t * \
            (1.0 / (self.material.specific_heat * self.material.density)) * \
            self.eff_visc(self.params.initial_stress, self.T_) * \
            self.params.initial_stress ** 2 * self.v * dfn.dx

        self.temp_form = self.diffusive_term + self.source_term + \
            self.shear_heat_term

        # J must be a Jacobian (Gateaux derivative in direction of du)
        self.temp_jac = dfn.derivative(self.temp_form, self.T_, self.dT)

    def eff_visc(self, stress, temp):
        return self.material.creep_constant * \
            stress ** (self.material.stress_exponent - 1) * \
            dfn.exp(-(self.material.activation_energy / consts.R) / temp)

    def _compute(self):
        dfn.set_log_level(16)

        # NonlinearVariationalProblem
        problem = NVP(self.temp_form, self.T_, self.bcT, self.temp_jac)
        # NonlinearVariationalSolver
        solver = NVS(problem)
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-8
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        dfn.info(prm, True)
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['preconditioner'] = 'ilu'
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
        prm['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
        t = self.params.delta_t
        t_max = self.params.t_max
        while t <= t_max:
            solver.solve()
            t += self.params.delta_t
            self.T_old.assign(self.T_)
            dfn.plot(self.T_)
            print(np.max(self.T_.vector().array()))
        pyp.show()

    def _visualize(self):
        pyp.plot(dfn.interpolate(self.initial_T, self.V).vector().array())
        pyp.plot(self.T_.vector().array())
        pyp.show()
        pyp.plot(self.s_.vector().array())
        pyp.show()
