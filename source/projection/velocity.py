from matplotlib import pyplot as pyp
import dolfin as dfn
import numpy as np
from core.debug import _DEBUG
from linear_viscoelastic.constant_slip_maxwell import velocity_solution

class BoundaryCondExpr(dfn.Expression):
    def __init__(self, params):
        self.t = 0
        self.params = params

    def eval(self, value, x):
        v_3 = 0.0#velocity_solution(x[0],
                                # x[1],
                                # self.t,
                                # self.params.t_r,
                                # self.params.fault_depth,
                                # self.params.elastic_depth,
                                # self.params.fault_slip)
        value[0] = v_3

class VelocitySolver(object):

    """
    Solves the poisson equation for velocity to maintain
    divergence free stress conditions.
    """
    tol = 1E-14   # tolerance for coordinate comparisons

    def vel_left_boundary(self, x, on_boundary):
        """ Identify left (west) boundary.  """
        return on_boundary and abs(x[0] - self.params.x_min) < self.tol

    def vel_right_boundary(self, x, on_boundary):
        """ Identify right (east) boundary.  """
        return on_boundary and abs(x[0] - self.params.x_max) < self.tol

    def vel_bottom_boundary(self, x, on_boundary):
        """ Identify bottom (mantle) boundary.  """
        return on_boundary and abs(x[1] - self.params.y_max) < self.tol

    def __init__(self, params, mesh):
        self.fnc_space = dfn.FunctionSpace(mesh, 'CG', 1)
        self.mesh = mesh
        self.params = params
        self.material = params.material

        # Create a grid for the velocity. Not necessary for computations,
        # just good to orient myself.
        self.velX_ = np.linspace(self.params.x_min,
                                 self.params.x_max,
                                 self.params.x_points)
        self.velY_ = np.linspace(self.params.y_min,
                                 self.params.y_max,
                                 self.params.y_points)
        self.velX, self.velY = np.meshgrid(self.velX_, self.velY_)
        # Setup the boundary conditions. Dirichlet on the west, east and bottom.
        # Neumann on the free surface (top). But, the natural Neumann boundary condition is 0
        # so it does not show up anywhere in the formulation
        # Western, south boundary condition
        self.bc_expr = BoundaryCondExpr(self.params)
        self.vel_bc_left = dfn.DirichletBC(
            self.fnc_space, self.bc_expr,
            lambda x, on_boundary:
            self.vel_left_boundary(x, on_boundary))

        # Eastern, right boundary condition
        self.vel_bc_right = dfn.DirichletBC(
            self.fnc_space, self.bc_expr,
            lambda x, on_boundary:
            self.vel_right_boundary(x, on_boundary))

        # Mantle boundary conditions
        self.vel_bc_bottom = dfn.DirichletBC(
            self.fnc_space, self.bc_expr,
            lambda x, on_boundary:
            self.vel_bottom_boundary(x, on_boundary))

        self.velocity = dfn.TrialFunction(self.fnc_space)
        self.velocity_test = dfn.TestFunction(self.fnc_space)

        a = dfn.inner(dfn.nabla_grad(self.velocity),
                      dfn.nabla_grad(self.velocity_test)) * dfn.dx
        self.vel_f = dfn.Function(self.fnc_space)
        # The divergence will be computed manually using a finite difference approximation.
        self.vel_L = self.vel_f * self.velocity_test * dfn.dx
        # assemble only once, before the time stepping
        self.vel_A = dfn.assemble(a)

    def convert_div_stress_to_fenics(self, factor, divS):
        divS_flat = factor * divS.flatten()
        self.vel_f.vector()[:] = divS_flat[self.fnc_space.
                                           dofmap().
                                           vertex_to_dof_map(self.mesh)]

        # test it
        re_extracted_orig = self.vel_f.vector()\
            [self.fnc_space.dofmap().dof_to_vertex_map(self.mesh)].\
            array().reshape((self.params.y_points,
                             self.params.x_points))
        assert(np.sum(np.sum(np.abs(re_extracted_orig - factor * divS))) == 0)

    def calc_div_stress(self, Szx, Szy):
        divS = np.zeros((self.params.x_points, self.params.y_points))
        divS[:, 1:-1] += (Szx[:, 1:] - Szx[:, :-1]) / self.params.delta_x
        divS[1:-1, :] += (Szy[1:, :] - Szy[:-1, :]) / self.params.delta_y
        return divS

    def update(self, t, dt, Szx, Szy):
        factor = 1 / (self.material.shear_modulus * dt)
        divS = self.calc_div_stress(Szx, Szy)
        self.convert_div_stress_to_fenics(factor, divS)
        # Complex reordering to get in the preferred FeNICs forms.
        # z = np.zeros((self.params.x_points * self.params.y_points, 1))

        b = dfn.assemble(self.vel_L)
        self.bc_expr.t = t
        self.vel_bc_left.apply(self.vel_A, b)
        # self.vel_bc_right.apply(self.vel_A, b)
        # self.vel_bc_bottom.apply(self.vel_A, b)
        u = dfn.Function(self.fnc_space)
        dfn.solve(self.vel_A, u.vector(), b)
        extracted_soln = u.vector()\
            [self.fnc_space.dofmap().dof_to_vertex_map(self.mesh)].\
            array().reshape((self.params.y_points, self.params.x_points))
        return extracted_soln
