from gryphon import ESDIRK
from dolfin import *

# Define spatial mesh, function space, trial/test functions
mesh = UnitSquare(29,29)
V = FunctionSpace(mesh,"Lagrange",1)
u = TrialFunction(V)
v = TestFunction(V)

# Define diffusion coefficient and source inside domain
D = Constant(0.1)
domainSource = Expression("10*sin(pi/2*t)*exp(-((x[0]-0.7)*(x[0]-0.7) + (x[1]-0.5)*(x[1]-0.5))/0.01)",t=0)

# Define right hand side of the problem
rhs = -D*inner(grad(u),grad(v))*dx + domainSource*v*dx

# Definie initial condition
W = Function(V)
W.interpolate(Constant(0.0))

# Define left and right boundary
def boundaryLeft(x,on_boundary):
  return x[0] < DOLFIN_EPS

def boundaryRight(x,on_boundary):
  return 1.0 - x[0] < DOLFIN_EPS

boundarySource = Expression("t",t=0)
bcLeft  = DirichletBC(V,boundarySource,boundaryLeft)
bcRight = DirichletBC(V,0.0,boundaryRight)

# Define the time domain
T = [0,1]

# Create the ESDIRK object
obj = ESDIRK(T,W,rhs,bcs=[bcLeft,bcRight],tdfBC=[boundarySource],tdf=[domainSource])

# Turn on some output and save run time
# statistics to sub folder "HeatEquation"
obj.parameters["verbose"] = True
obj.parameters["drawplot"] = True
obj.parameters["output"]["path"] = "HeatEquation"
obj.parameters["output"]["statistics"] = True

# Solve the problem
obj.solve()
