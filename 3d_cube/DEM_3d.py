#coding: utf-8

# Plot the error in SCF depending on the elements size
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *

# Parameters
R = 0.5e-2 # radius
cube = 5e-2 # dim

#compressible
T = 5e9 # traction force
nu = 0.499999 # Poisson's ratio
E = 3e9 #Young Modulus
G = 0.5*E/(1+nu) #Shear modulus
Gc = 0.84e9 # Second shear modulus
lmbda = 2*G*nu / (1-2*nu) # 1st Lame constant
l = 1e-3 #intrinsic length
M = 2*G*l*l
L = 2/3*M
Mc = M

#Loading mesh
mesh_num = 10
mesh = BoxMesh(Point(0., 0., 0.), Point(cube, cube, cube), mesh_num, mesh_num, mesh_num)

#Creating the DEM problem
cte = 100
problem = DEMProblem(mesh, cte)
print('nb dofs: %i' % problem.nb_dof_DEM)

#Computing coefficients for Cosserat material
problem.micropolar_constants_3d(E, nu, Gc, L, M, Mc)

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

boundary_parts = MeshFunction("size_t", mesh, problem.dim-1)
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
A += inner_penalty(problem) #light

#rhs
t = Constant((0.0, T, 0.0))
b = assemble_boundary_load(problem, 1, boundary_parts, t)
#Imposing weakly the BC!
#b += rhs_bnd_penalty(problem, boundary_parts, bcs)

v_DG = Function(problem.V_DG)
print('Solve!')
solve(PETScMatrix(A), v_DG.vector(), PETScVector(b), 'mumps')
u_DG, phi_DG = v_DG.split()
v_DG1 = Function(problem.V_DG1)
v_DG1.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
u_DG1, phi_DG1 = v_DG1.split()

file = File('DEM/locking_%i_.pvd' % mesh_num)
#file << u_DG
file << u_DG1
#file << phi_DG
file << phi_DG1

epsilon_u_h = problem.strain_3d(u_DG1, phi_DG1)
sigma_u_h = problem.stress_3d(epsilon_u_h)
U = FunctionSpace(problem.mesh, 'DG', 0)
#U = FunctionSpace(problem.mesh, 'CG', 1)
sigma_yy = local_project(sigma_u_h[1,1], U)
file << sigma_yy
sys.exit()

#Comparing SCF
SCF = sigma_yy(R, 0, 0)
e = abs(SCF - SCF_a) / SCF_a
print('Ref: %.5e' % SCF_a)
print('Computed: %.5e' % SCF)
print('Error: %.2f' % (100*e))
sys.exit()

#Computing CG ref solution
u_ref,phi_ref = computation(mesh, cube, T, nu, G, Gc, l, mesh_num)

#Computing errors
err_L2_u = errornorm(u_DG1, u_ref, 'L2') #, degree_rise=0)
print(err_L2_u)
err_L2_phi = errornorm(phi_DG1, phi_ref, 'L2') #, degree_rise=0)
print(err_L2_phi)
tot_l2 = np.sqrt(err_L2_u**2+err_L2_phi**2)
print('Tot L2: %.3e' % tot_l2)

err_H10_u = errornorm(u_DG1, u_ref, 'H10') #, degree_rise=0)
print(err_H10_u)
err_H10_phi = errornorm(phi_DG1, phi_ref, 'H10') #, degree_rise=0)
print(err_H10_phi)
tot_H10 = np.sqrt(err_H10_u**2+err_H10_phi**2)
print('Tot H10: %.3e' % tot_H10)
