from gryphon import ESDIRK
from dolfin import *
import random

# Initial conditions
class InitialConditions(Expression):
  def __init__(self):
      random.seed(2 + MPI.process_number())
  def eval(self, values, x):
      values[0] = 0.63 + 0.02*(0.5 - random.random())
      values[1] = 0.0
  def value_shape(self):
      return (2,)

# Create mesh and define function spaces
mesh = UnitSquare(49, 49)
V = FunctionSpace(mesh, "Lagrange", 1)
ME = V*V

q,v = TestFunctions(ME)

# Define and interpolate initial condition
u   = Function(ME)
u.interpolate(InitialConditions())

c,mu = split(u)
c = variable(c)
f    = 100*c**2*(1-c)**2
dfdc = diff(f, c)
lmbda  = Constant(1.0e-02)

# Weak statement of the equations
f = -inner(grad(mu), grad(q))*dx
g = mu*v*dx - dfdc*v*dx - lmbda*inner(grad(c), grad(v))*dx

T = [0,5e-5] # Time domain

myobj = ESDIRK(T,u,f,g=g)
myobj.parameters['timestepping']['absolute_tolerance'] = 1e-2
myobj.parameters['timestepping']['inconsistent_initialdata'] = True
myobj.parameters['verbose'] = True
myobj.parameters['drawplot'] = True

# Suppress some FEniCS output
set_log_level(WARNING)

# Solve the problem
myobj.solve()
