from matplotlib import pyplot as pyp
import dolfin as dfn
import numpy as np
from core.debug import _DEBUG


class VelocitySolver():

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

    def __init__(self, params, func_space, mesh):
        self.fnc_space = func_space
        self.vec_fnc_space = dfn.VectorFunctionSpace(mesh, 'CG', 1)
        self.mesh = mesh
        self.params = params
        self.material = params.material
        # Setup the boundary conditions. Dirichlet on the west, east and bottom.
        # Neumann on the free surface (top). But, the natural Neumann boundary condition is 0
        # so it does not show up anywhere in the formulation
        # Western, south boundary condition
        self.vel_bc_left = dfn.DirichletBC(
            func_space, dfn.Constant(self.params.plate_rate),
            lambda x, on_boundary:
            self.vel_left_boundary(x, on_boundary))

        # Eastern, right boundary condition
        self.vel_bc_right = dfn.DirichletBC(
            func_space, dfn.Constant(-self.params.plate_rate),
            lambda x, on_boundary:
            self.vel_right_boundary(x, on_boundary))

        # Mantle boundary conditions
        self.vel_bc_bottom = dfn.DirichletBC(func_space, dfn.Constant(0.00),
                                             lambda x, on_boundary:
                                             self.vel_bottom_boundary(x, on_boundary))
        self.velocity = dfn.TrialFunction(func_space)
        self.velocity_test = dfn.TestFunction(func_space)

        a = dfn.inner(dfn.nabla_grad(self.velocity),
                      dfn.nabla_grad(self.velocity_test)) * dfn.dx
        self.vel_f = dfn.Function(self.vec_fnc_space)
        self.vel_L = dfn.nabla_div(self.vel_f) * self.velocity_test * dfn.dx
        # assemble only once, before the time stepping
        self.vel_A = dfn.assemble(a)

    def update(self, t, dt, Szx, Szy):
        factor = 1 / (self.material.shear_modulus * dt)
        # Complex reordering to get in the preferred FeNICs forms.
        Szx_flat = factor * Szx.reshape((self.params.x_points * self.params.y_points, 1))
        Szy_flat = factor * Szy.reshape((self.params.x_points * self.params.y_points, 1))
        # z = np.zeros((self.params.x_points * self.params.y_points, 1))
        array = np.concatenate((Szx_flat, Szy_flat), 1).flatten()
        _DEBUG(1)
        self.vel_f.vector()[:] = array[
            self.vec_fnc_space.dofmap().vertex_to_dof_map(self.mesh)]
        dfn.plot(self.vel_f)

        b = dfn.assemble(self.vel_L)
        self.vel_bc_left.apply(self.vel_A, b)
        self.vel_bc_right.apply(self.vel_A, b)
        self.vel_bc_bottom.apply(self.vel_A, b)
        u = dfn.Function(self.fnc_space)
        dfn.solve(self.vel_A, u.vector(), b)
        extracted_soln = u.vector()[self.fnc_space.dofmap().dof_to_vertex_map(self.mesh)].\
            array().reshape((self.params.x_points, self.params.y_points))
        pyp.imshow(extracted_soln)
        pyp.show()
