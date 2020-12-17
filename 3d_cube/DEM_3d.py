#coding: utf-8

# Plot the error in SCF depending on the elements size
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from solver_3D import computation

# Parameters
R = 10.0 # radius
cube = 100.0 # dim

T = 1.0 # traction force

nu = 0.3 #0.49 #0.3 # Poisson's ratio
G = 10e6 # shear modulus
Gc = 5e6 #other shear modulus
E = 2*G*(1+nu) #Yound Modulus
#lmbda = ( 2.*mu*nu ) / (1-2*nu) # 1st Lame constant

l = 10 # intrinsic length scale
#N = 0.93 # coupling parameter
h3 = 2/5
M = G * l*l/h3

#Loading mesh
mesh = Mesh()
mesh_num = 1
with XDMFFile("meshes/cube_%i.xdmf" % mesh_num) as infile:
    infile.read(mesh)
hm = mesh.hmax()
print(hm)

#Creating the DEM problem
cte = 10
problem = DEMProblem(mesh, cte)
print('nb dof DEM: %i' % problem.nb_dof_DEM)

#Computing coefficients for Cosserat material
problem.micropolar_constants(E, nu, l, Gc, M)

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

#test
from petsc4py import PETSc
A = A.tocsr()
petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices,A.data))
A_aux = PETScMatrix(petsc_mat)
x = Function(problem.V_DG)
x.vector().set_local(b)
xx = x.vector()
v_DG = Function(problem.V_DG)
print('Solve!')
solve(A_aux, v_DG.vector(), xx, 'mumps')

u_DG, phi_DG = v_DG.split()
v_DG1 = Function(problem.V_DG1)
v_DG1.vector().set_local(problem.DEM_to_DG1 * v_DG.vector())
u_DG1, phi_DG1 = v_DG1.split()

file = File('no_locking_%i_.pvd' % mesh_num)
file << u_DG
file << u_DG1
file << phi_DG
file << phi_DG1

#epsilon_u_h = problem.strain_3d(u_DG1, phi_DG1)
#sigma_u_h = problem.stress_3d(epsilon_u_h)
##U = FunctionSpace(problem.mesh, 'DG', 0)
#U = FunctionSpace(problem.mesh, 'CG', 1)
#sigma_yy = local_project(sigma_u_h[1,1], U)
#file << sigma_yy
#
##Comparing SCF
#SCF = sigma_yy(R, 0, 0)
#e = abs(SCF - SCF_a) / SCF_a
#print('Ref: %.5e' % SCF_a)
#print('Computed: %.5e' % SCF)
#print('Error: %.2f' % (100*e))

#Computing errors
u_ref,phi_ref = computation(mesh, R, cube, T, nu, mu, lmbda, l, N)
err_L2_u = errornorm(u_DG1, u_ref, 'L2', degree_rise=0)
#print(err_L2_u)
err_L2_phi = errornorm(phi_DG1, phi_ref, 'L2', degree_rise=0)
#print(err_L2_phi)
tot_l2 = np.sqrt(err_L2_u**2+err_L2_phi**2)
print('Tot L2: %.3e' % tot_l2)

err_H10_u = errornorm(u_DG1, u_ref, 'H10', degree_rise=0)
print(err_H10_u)
err_H10_phi = errornorm(phi_DG1, phi_ref, 'H10', degree_rise=0)
print(err_H10_phi)
tot_H10 = np.sqrt(err_H10_u**2+err_H10_phi**2)
print('Tot H10: %.3e' % tot_H10)
