#coding: utf-8

# Plot the error in SCF depending on the elements size
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve,cg

# Parameters
R = 10.0 # radius
cube = 100.0 # dim

T = 1.0 # traction force

nu = 0.3 #0.49 #0.3 # Poisson's ratio
mu = 1000.0 # shear modulus G
lmbda = ( 2.0 * mu * nu ) / (1.0-2.0*nu) # 1st Lame constant

l = 0.2 # intrinsic length scale
N = 0.93 # coupling parameter

# Analytical solution
def AnalyticalSolution(R, l, nu):
    k = R / l
    eta = 0.2 # ratio of the transverse curvature to the principal curvature
    k_1 = (3.0+eta) / ( 9.0 + 9.0*k + 4.0*k**2 + eta*(3.0 + 3.0*k + k**2) )
    SCF = ( 3.0*(9.0 - 5.0*nu + 6.0*k_1*(1.0-nu)*(1.0+k)) ) / \
          ( 2.0*(7.0 - 5.0*nu + 18.0*k_1*(1.0-nu)*(1.0+k)) )
    return SCF

SCF_a = AnalyticalSolution(R, l, nu)

Loading mesh
mesh = Mesh("meshes/3.xml")
hm = mesh.hmax()

#Creating the DEM problem
problem = DEMProblem(mesh, 4*G, 4*G*l*l)
print('nb dof DEM: %i' % problem.nb_dof_DEM)

# Boundary conditions
# Boundary conditions
    class BotBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1]) < tol

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[0]) < tol

class FrontBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[2]) < tol

class TopBoundary(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1] - cube) < tol

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
        
bot_boundary = BotBoundary()
left_boundary = LeftBoundary()
front_boundary = FrontBoundary()
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

#Computation
SCF_0 = computation(mesh, R, cube, T, nu, mu, lmbda, l, N)

#Comparing SCF
e = abs(SCF_0 - SCF_a) / SCF_a
print('Error: %.5e' % e)
