import dolfin as dfn
import numpy as np
# from math import exp
from matplotlib import pyplot as pyp
# from pdb import set_trace as _DEBUG
# import ufl.operators
from core.experiment import Experiment
from core.constants import consts


class ShearHeatingFenics(Experiment):
    """
    Solving the shear heating equations with FEnICS.
    This is entirely playcode. Don't use it without
    serious refactoring and testing.
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

        class SourceTermExpr(dfn.Expression):
            def __init__(self, src, ls):
                self.src = src
                self.ls = ls

            def eval(self, value, x):
                value[0] = 0.0
                if abs(x[0]) <= (1.0 / self.ls):
                    value[0] = self.src
        source = SourceTermExpr(self.data.source_term, self.data.length_scale)

        # Create mesh and define function space
        self.mesh = dfn.RectangleMesh(self.data.x_min, self.data.x_min, self.data.x_max, self.data.x_max, self.params.x_points, self.params.x_points)
        self.V = dfn.FunctionSpace(self.mesh, 'CG', 1)
        self.ME = self.V * self.V

        # Define boundary conditions -- gaussian initial conditions
        W = dfn.Function(self.ME)
        W.interpolate(dfn.Constant((0.0, 0.0)))
        self.u, self.s = dfn.split(W)
        self.bc = dfn.DirichletBC(self.ME, dfn.Constant((0.0, 0.0)), lambda x, on_bndry: on_bndry)

        # Define variational problem
        self.v1, self.v2 = dfn.TestFunctions(self.ME)
        self.temp_RHS = - dfn.inner(dfn.nabla_grad(self.u + self.s), dfn.nabla_grad(self.v1)) * dfn.dx + \
            source * self.v1 * dfn.dx

        self.shear_heat_RHS = self.data.inv_prandtl * self.data.eckert * \
            self.data.stress ** (self.material.stress_exponent + 1) * \
            dfn.exp(-(self.material.activation_energy / consts.R) /
                    (self.params.low_temp + self.params.delta_temp * (self.u + self.s))) * \
            self.v2 * dfn.dx

        T = [0, self.data.t_max]
        # To  make this work, reimplement the time stepping in the compute section below
        # self.solver = ESDIRK(T, W, [self.temp_RHS, self.shear_heat_RHS], bcs=[self.bc])
        # self.solver.is_linear = False
        # self.solver.parameters['timestepping']['dt'] = self.data.delta_t
        # self.solver.parameters["timestepping"]["absolute_tolerance"] = 1e-7


    def _compute(self):
        self.solver.parameters["verbose"] = True
        self.solver.parameters["drawplot"] = True
        self.solver.parameters['output']['statistics'] = False
        # self.solver.parameters["output"]["path"] = "GrayScott"

        # Supress some FEniCS output
        # dfn.set_log_level(dfn.WARNING)

        # Solve the problem
        self.data.soln = self.solver.solve()

    def _visualize(self):
        pass
        # pyp.plot(self.u_.vector().array())
        # pyp.show()
