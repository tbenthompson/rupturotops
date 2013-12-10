import dolfin as dfn


class SourceTermExpr(dfn.Expression):

    def __init__(self, src):
        self.src = src

    def eval(self, value, x):
        value[0] = 0.0
        value[0] = self.src

class TempSolver(object):
    def __init__(self, params, mesh):
        self.params = params
        self.material = params.material
        self.mesh = mesh

        # First order Lagrange triangles.
        self.fnc_space = dfn.FunctionSpace(self.mesh, 'CG', 1)

        self.initial_temp = dfn.Expression('A + B * exp(-x[0] * x[0] / 100)',
                                           A=self.params.background_temp,
                                           B=self.params.temp_pulse_size)

        self.bctemp = dfn.DirichletBC(self.fnc_space,
                                      self.initial_temp,
                                      lambda x, on_bndry: on_bndry)

        # source = SourceTermExpr(self.params.source_term)
        source = dfn.Expression('B * exp(-x[0] * x[0])',
                                B=self.params.source_term)

        # initial conditions
        self.temp_ = dfn.interpolate(self.initial_temp, self.fnc_space)
        self.temp_old = dfn.interpolate(self.initial_temp, self.fnc_space)

        # Define variational problem
        self.dtemp = dfn.TrialFunction(self.fnc_space)
        self.temp_test = dfn.TestFunction(self.fnc_space)
        self.diffusive_term = self.params.delta_t * \
            self.material.thermal_diffusivity * \
            dfn.inner(dfn.nabla_grad(self.temp_),
                      dfn.nabla_grad(self.temp_test)) * dfn.dx + \
            self.temp_ * self.temp_test * dfn.dx - \
            self.temp_old * self.temp_test * dfn.dx

        self.source_term = -self.params.delta_t * \
            (1.0 / (self.material.specific_heat * self.material.density)) * \
            source * self.temp_test * dfn.dx

        self.shear_heat_term = self.params.delta_t * \
            (1.0 / (self.material.specific_heat * self.material.density)) * \
            self.params.initial_stress ** 2 * self.temp_test * dfn.dx
            # self.inv_eff_visc(self.params.initial_stress, self.temp_) * \

        self.temp_form = self.diffusive_term + self.source_term + \
            self.shear_heat_term

        # J must be a Jacobian (Gateaux derivative in direction of du)
        self.temp_jac = dfn.derivative(self.temp_form, self.temp_, self.dtemp)
        # NonlinearVariationalProblem
        self.temp_problem = NVP(
            self.temp_form, self.temp_, self.bctemp, self.temp_jac)
        # NonlinearVariationalSolver
        self.temp_solver = NVS(problem)
        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-8
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        prm['newton_solver']['linear_solver'] = 'gmres'
        prm['newton_solver']['preconditioner'] = 'ilu'
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
        prm['newton_solver']['krylov_solver'][
            'preconditioner']['ilu']['fill_level'] = 0

    def update_temp(self):
        pass
        # solver.solve()
        # self.temp_old.assign(self.temp_)
        # dfn.plot(self.temp_)
        # print(np.max(self.temp_.vector().array()))

