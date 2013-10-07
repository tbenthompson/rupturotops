import dolfin as dfn
# import numpy
# from math import exp
from matplotlib import pyplot as pyp
from pdb import set_trace as _DEBUG
# import ufl.operators

# Create mesh and define function space
mesh = dfn.UnitIntervalMesh(50)
V = dfn.FunctionSpace(mesh, 'Lagrange', 2)

# Define boundary conditions -- gaussian initial conditions
u0 = dfn.Expression('(1/sqrt(3.1415)) * exp(-((x[0]-0.5)*(x[0]-0.5))*5)')


def u0_boundary(x, on_boundary):
    return on_boundary

bc = dfn.DirichletBC(V, u0, u0_boundary)

deltaT = 0.001

P = 0.1


# Define variational problem
v = dfn.TestFunction(V)
du = dfn.TrialFunction(V)
u_ = dfn.Function(V)  # most recently computed solution
u_ = dfn.interpolate(u0, V)
pyp.plot(u_.vector().array())
F = dfn.inner(dfn.nabla_grad(u_), dfn.nabla_grad(v)) * dfn.dx + \
    P * dfn.exp(-1 / u_) * v * dfn.dx
# J must be a Jacobian (Gateaux derivative in direction of du)
J = dfn.derivative(F, u_, du)

# initial conditions
problem = dfn.NonlinearVariationalProblem(F, u_, bc, J)
solver = dfn.NonlinearVariationalSolver(problem)
_DEBUG()

prm = solver.parameters
#info(prm, True)
prm['linear_solver'] = 'gmres'
prm['preconditioner'] = 'ilu'
prm['krylov_solver']['absolute_tolerance'] = 1E-9
prm['krylov_solver']['relative_tolerance'] = 1E-7
prm['krylov_solver']['maximum_iterations'] = 1000
prm['krylov_solver']['gmres']['restart'] = 40
prm['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
dfn.set_log_level(16)
solver.solve()
#
#     t += deltaT
#     u_1.assign(u)
#     # _DEBUG()
pyp.plot(u_.vector().array())
