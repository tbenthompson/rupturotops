# Copyright (C) 2012 - Knut Erik Skare
#
# This file is part of Gryphon.
#
# Gryphon is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Gryphon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Gryphon. If not, see <http://www.gnu.org/licenses/>.

from dolfin import Constant,NewtonSolver,TestFunction,TrialFunction,Parameters
from dolfin import LUSolver,KrylovSolver,info,Function,DirichletBC
from ufl import form
from os import makedirs

class gryphon_base():
  def __init__(self,T,u,f,bcs=[],tdf=[],tdfBC=[]):
    self.u = u                                        # Initial condition
    self.bcs = bcs                                    # Boundary conditions
    self.f = f                                        # Right hand side
    self.tdf = tdf                                    # Time dependent functions
    self.tdfBC = tdfBC                                # Time dependent functions on the boundary
    self.T = T                                        # Time domain
    self.verifyInput()
    self.t = T[0]                                     # Current time
    self.tstart = T[0]                                # Start of domain
    self.tend = T[len(T)-1]                           # End of domain
    self.dt = (self.tend-self.tstart)/1000.0          # Initial time step
    self.Q = TestFunction(self.u.function_space())    # Trial Function to right hand side
    self.U = TrialFunction(self.u.function_space())   # Test Function to right hand side
    self.DT = Constant(self.dt)                       # Current step size, used in forms

    # Determine size and rank of system
    if type(self.f) == type([]):
        self.n = len(self.f)
        self.rank = self.f[0].compute_form_data().rank
    else:
        self.n = 1
        self.rank = self.f.compute_form_data().rank

    # Set appropriate solver
    if self.rank==2:
        info("Using LU-solver to solve linear systems.")
        self.linear = True
        self.solver = LUSolver()
    else:
        info("Using Newton-solver to solve nonlinear systems.")
        self.linear = False
        self.solver = NewtonSolver()

  def odeError(self,task,reason,remedy=False):
    EM =  "\n----------------------------------------------------------------------------"
    EM += "\n                *** The ODE solver encountered an error ***"
    EM += "\n----------------------------------------------------------------------------"
    EM += "\n*** " + "Error : " + task
    EM += "\n*** " + "Reason: " + reason
    if remedy:
      EM += "\n*** " + "Remedy: " + remedy
    EM += "\n----------------------------------------------------------------------------"
    EM += "\n"
    raise RuntimeError(EM)

  def ensureFolder(self):
    try:
        makedirs(self.parameters['output']['path'])
    except OSError:
        pass

  # ------------------------------------------------------------------

  def printProgress(self,timeEstimate=False,rejectedStep=False):
    n = 20    # Width of progress bar
    totalTime = self.tend - self.tstart
    elapsedTime = self.t - self.tstart
    progress = round(100.0*elapsedTime/totalTime,1)
    dots = int(round(n*progress/100.0))

    progressBar = "|" + "="*dots + ">" + "."*(n-dots) + "| %s%%  \tt=%g" %(progress,self.t)
    if rejectedStep:
      progressBar += "\tStep rejected, dt=%g" % self.dt

    if timeEstimate and (self.tstart+self.t)/(self.tend-self.tstart) > 0.01 and not rejectedStep:
      progressBar += "\tCompletion in ~ %s" %(timeEstimate())

    print progressBar

  def verifyInput(self):
    if type(self.T) != list:
      self.odeError("Error in first required argument: Time domain.",
                     "Domain must be speficied as a list with two floats representing t_start and t_end.")
    elif self.T[1] - self.T[0] < 0:
      self.odeError("Error in first required argument: Time domain.",
                     "Specified time domain [%s,%s] is negative."%(self.T[0],self.T[1]))

    if type(self.u) != Function:
      self.odeError("Error in second required argument: Initial condition.",
                     "The initial condition must be given as a Function-object.")

    if type(self.f) != list:
      self.f = [self.f]
    if any([type(self.f[i]) != form.Form for i in range(len(self.f))]):
      self.odeError("Error in third required argument: Right hand side.",
                     "The right hand side must be given as either a single Form-object or a list of Form-objects.")

    if self.bcs != []:
      if type(self.bcs) != list:
        self.bcs = [self.bcs]
      if any([type(self.bcs[i]) != DirichletBC for i in range(len(self.bcs))]):
        self.odeError("Error in keyword argument: 'bcs'.",
                       "Boundary conditions must be given as either a single DirichletBC object" +
                       "\n            or a list of DirichletBC objects.")

    if self.tdf != []:
      if type(self.tdf) != list:
        self.tdf = [self.tdf]

    if self.tdfBC != []:
      if type(self.tdfBC) != list:
        self.tdf = [self.tdfBC]

  def printVersion(self):
    print "You are currently using version 1.0 of Gryphon."
