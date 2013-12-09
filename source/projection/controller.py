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


class SourceTermExpr(dfn.Expression):

    def __init__(self, src):
        self.src = src

    def eval(self, value, x):
        value[0] = 0.0
        value[0] = self.src


class ProjController(Experiment):

    """
    Solving the shear heating equations with FEnICS
    """

    def _initialize(self):
        # Create mesh and define function space
        # Subtract one from points to get the number of elements
        self.mesh = dfn.RectangleMesh(self.params.x_min, self.params.y_min,
                                      self.params.x_max, self.params.y_max,
                                      self.params.x_points - 1,
                                      self.params.y_points - 1)
        # First order Lagrange triangles.
        self.fnc_space = dfn.FunctionSpace(self.mesh, 'CG', 1)

        self.setup_momentum_eqtn()
        self.vel_solver = VelocitySolver(self.params, self.fnc_space, self.mesh)
        # self.setup_energy_eqtn()

    def elastic_stress_soln(self, x, y, s, D):
        """
        Use the elastic half-space stress solution from Segall (2010)
        """
        # TEST THIS!
        factor = (s * self.material.shear_modulus) / (2 * np.pi)
        main_term = -(y - D) / ((y - D) ** 2 + x ** 2)
        image_term = (y + D) / ((y + D) ** 2 + x ** 2)
        sigma_zx = factor * (main_term + image_term)

        main_term = x / (x ** 2 + (y - D) ** 2)
        image_term = -x / (x ** 2 + (y + D) ** 2)
        sigma_zy = factor * (main_term + image_term)
        # pyp.figure(1)
        # pyp.imshow(np.log(np.abs(sigma_zy)))
        # pyp.colorbar()
        # pyp.figure(2)
        # pyp.imshow(np.log(np.abs(sigma_zx)))
        # pyp.colorbar()
        # pyp.show()
        return sigma_zx, sigma_zy

    def setup_momentum_eqtn(self):
        self.X_ = np.linspace(
            self.params.x_min, self.params.x_max, self.params.x_points)
        self.Y_ = np.linspace(
            self.params.y_min, self.params.y_max, self.params.y_points)
        self.X, self.Y = np.meshgrid(self.X_, self.Y_)
        self.Szx, self.Szy = self.elastic_stress_soln(
            self.X, self.Y, self.params.fault_slip, self.params.fault_depth)
        self.initialSzx = self.Szx.copy()
        self.initialSzy = self.Szy.copy()
        # pyp.imshow(np.log(self.stress))
        # pyp.show()
        # backend = 'dopri5'
        # self.stress_solver = scipy.integrate.ode(self.deriv_momentum).\
        #     set_integrator(backend, nsteps=1, first_step=self.params.delta_t,
        #                    verbosity=100)
        # suppress Fortran-printed warning
        # self.stress_solver._integrator.iwork[2] = -1

    def deriv_momentum(self, t, Szx, Szy):
        factor = -(self.material.shear_modulus *
                   self.inv_eff_visc(self.params.initial_stress, 0.0,
                                     self.params.background_temp))
        dSzx = factor * Szx
        dSzy = factor * Szy
        return dSzx, dSzy

    def eff_stress(self, Szx, Szy):
        return np.sqrt(Szx ** 2 + Szy ** 2)

    def inv_eff_visc(self, Szx, Szy, temp):
        retval = 1.0 / 5e19
        # retval = self.material.creep_constant * \
        #     self.eff_stress(Szx, Szy) ** (self.material.stress_exponent - 1) * \
        #     dfn.exp(-(self.material.activation_energy / consts.R) / temp)
        return retval

    def update_momentum(self, t, dt):
        # self.stress_solver.set_initial_value(self.stress, t)
        sol = []
        # warnings.filterwarnings("ignore", category=UserWarning)
        internal_t = t
        internal_dt = dt / 100.0
        while internal_t < t + dt:
            dSzx, dSzy = self.deriv_momentum(
                internal_t, self.Szx, self.Szy)
            self.Szx += internal_dt * dSzx
            self.Szy += internal_dt * dSzy
            internal_t += internal_dt
            # sol.append([self.stress_solver.t, self.stress_solver.y])
            # print("Time: " + str(internal_t))
            # print("Stress: " + str(deltaSzx))
        # self.stress = sol[-1][1]
        # warnings.resetwarnings()

    def setup_energy_eqtn(self):
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

    def _compute(self):
        dfn.set_log_level(16)

        dt = self.params.delta_t
        t = dt
        t_max = self.params.t_max
        pyp.figure(1)
        pyp.plot(self.X_, self.Szx[25, :])
        pyp.figure(2)
        pyp.plot(self.Y_, self.Szy[25, :])
        while t <= t_max:
            self.update_momentum(t, dt)
            # pyp.imshow(self.Szy)
            # pyp.show()
            dvdx, dvdy = self.vel_solver.update(t, dt, self.Szx, self.Szy)
            # pyp.plot(self.X_, self.Szx[25, :] + self.Szx_mod[25, :])
            # pyp.show()
            self.Szx += self.material.shear_modulus * dt * dvdx
            self.Szy += self.material.shear_modulus * dt * dvdy
            print np.mean(np.mean(self.Szx))
            # pyp.plot(self.X_, dvdx[25, :])
            # pyp.show()
            # pyp.plot(self.X_, dvdy[25, :])
            # pyp.show()
            # pyp.imshow(self.Szx_mod)
            # pyp.show()
            # pyp.show()
            # solver.solve()
            # self.temp_old.assign(self.temp_)
            # dfn.plot(self.temp_)
            # print(np.max(self.temp_.vector().array()))
            t += dt
            pyp.figure(1)
            pyp.plot(self.X_, self.Szx[25, :])
            pyp.figure(2)
            pyp.plot(self.Y_, self.Szy[25, :])
        pyp.show()

    def _visualize(self):
        # pyp.plot(dfn.interpolate(self.initial_temp, self.fnc_space).vector().array())
        from linear_viscoelastic.constant_slip_maxwell import solution
        beta = self.params.shear_modulus * (1.0 / 5e19)
        u_3 = solution(self.X, self.params.t_max / beta, self.params.fault_depth,
                       self.params.fault_depth, self.params.fault_slip)
        _DEBUG()
        # pyp.plot(self.temp_.vector().array())
        pyp.imshow(np.log(self.Szx))
        pyp.show()
