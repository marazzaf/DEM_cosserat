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
lmbda = ( 2.*mu*nu ) / (1-2*nu) # 1st Lame constant

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

#Loading mesh
mesh = Mesh("meshes/2.xml")
hm = mesh.hmax()

#Creating the DEM problem
problem = DEMProblem(mesh, 4*mu, 4*mu*l*l)
print('nb dof DEM: %i' % problem.nb_dof_DEM)

#Computing coefficients for Cosserat material
problem.micropolar_constants(nu, mu, lmbda, l, N)

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
bot_boundary.mark(boundary_parts, 3)
left_boundary = LeftBoundary()
left_boundary.mark(boundary_parts, 2)
front_boundary = FrontBoundary()
front_boundary.mark(boundary_parts, 4)
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

u_0 = Constant(0.0)

left_U_1 = [0, u_0, 2]
left_S_2 = [4, u_0, 2]
left_S_3 = [5, u_0, 2]

bot_U_2 = [1, u_0, 3]
bot_S_1 = [3, u_0, 3]
bot_S_3 = [5, u_0, 3]

front_U_3 = [2, u_0, 4]
front_S_1 = [3, u_0, 4]
front_S_2 = [4, u_0, 4]

bcs = [left_U_1, left_S_2, left_S_3, bot_U_2, bot_S_1, bot_S_3, front_U_3, front_S_1, front_S_2]

# Variational problem
A = problem.elastic_bilinear_form()
#Nitsche penalty bilinear form
A += lhs_bnd_penalty(problem, boundary_parts, bcs)
#Penalty matrix
A += inner_penalty_light(problem)

#rhs
t = Constant((0.0, T, 0.0))
b = assemble_boundary_load(problem, 1, boundary_parts, t)
#Imposing weakly the BC!
b += rhs_bnd_penalty(problem, boundary_parts, bcs)

#Solving linear problem
v = spsolve(A,b)
#v,info = cg(A,b)
#assert info == 0
v_DG = Function(problem.V_DG)
v_DG.vector().set_local(v)
u_DG, psi_DG = v_DG.split()
v_DG1 = Function(problem.V_DG1)
v_DG1.vector().set_local(problem.DEM_to_DG1 * v)
u_DG1, psi_DG1 = v_DG1.split()

file = File('3d.pvd')
file << u_DG
file << phi_DG

epsilon_u_h = strain(u_h, phi_h)
sigma_u_h = stress(lmbda, mu, kappa, epsilon_u_h)
sigma_yy = project(sigma_u_h[1,1])
SCF = sigma_yy(R, 0.0, 0.0)

#Comparing SCF
e = abs(SCF - SCF_a) / SCF_a
print('Error: %.5e' % e)
