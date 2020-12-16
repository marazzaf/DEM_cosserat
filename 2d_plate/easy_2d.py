#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from ref_2d import computation

# Parameters
nu = 0.3 # Poisson's ratio
G = 1000.0 # shear modulus
Gc = G
E = 2*G*(1+nu)
l = 0.2 # intrinsic length scale
T = 1.0 # load
    
# Mesh
mesh = Mesh()
with XDMFFile("hole_plate.xdmf") as infile:
#with XDMFFile("hole_plate_fine.xdmf") as infile:
#with XDMFFile("hole_plate_very_fine.xdmf") as infile:
    infile.read(mesh)

#Creating the DEM problem
#Creating the DEM problem
cte = 10 #10
problem = DEMProblem(mesh, cte) #1e3 semble bien
print('nb dof DEM: %i' % problem.nb_dof_DEM)

# Boundary conditions
class BotBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1]) < tol

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[0]) < tol

class TopBoundary(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1] - 100) < tol

t = Constant((0.0, T))
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

#For BC
bot_boundary = BotBoundary()
bot_boundary.mark(boundary_parts, 2)
left_boundary = LeftBoundary()
left_boundary.mark(boundary_parts, 3)
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

bc = [[0, Constant(0), 3], [2, Constant(0), 3], [1, Constant(0), 2], [2, Constant(0), 2]]

#For compliance tensor
problem.micropolar_constants(E, nu, l)

# Variational problem
A = problem.elastic_bilinear_form()

#rhs
b = assemble_boundary_load(problem, 1, boundary_parts, t)

#Imposing weakly the BC!
b += rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
A += lhs_bnd_penalty(problem, boundary_parts, bc)

#Penalty matrix
A += inner_penalty(problem) #light
#A += inner_consistency(problem)

#Converting matrix
from petsc4py import PETSc
A = A.tocsr()
petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices,A.data))
A = PETScMatrix(petsc_mat)
bb = Function(problem.V_DG)
bb.vector().set_local(b)
b = bb.vector()

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(A, v_DG.vector(), b, 'mumps')
u_DG,phi_DG = v_DG.split()
vec_DG = v_DG.vector().get_local()

#Computing reconstruction
v_DG1 = Function(problem.V_DG1)
v_DG1.vector().set_local(problem.DEM_to_DG1 * vec_DG)
u_DG1,phi_DG1 = v_DG1.split()

fig = plot(u_DG1[0])
plt.colorbar(fig)
plt.savefig('DEM/u_x.pdf')
plt.show()
fig = plot(u_DG1[1])
plt.colorbar(fig)
plt.savefig('DEM/u_y.pdf')
plt.show()
fig = plot(phi_DG1)
plt.colorbar(fig)
plt.savefig('DEM/phi.pdf')
plt.show()

#Computing errors
v_ref = computation(mesh, T, E, nu, l)
#err_L2_u = errornorm(u_DG1, u_ref, 'L2')#, degree_rise=0)
##print(err_L2_u)
#err_L2_phi = errornorm(phi_DG1, phi_ref, 'L2')#, degree_rise=0)
##print(err_L2_phi)
#tot_l2 = np.sqrt(err_L2_u**2+err_L2_phi**2)
err_L2 = errornorm(v_DG1, v_ref, 'L2')
print('Error L2: %.3e' % err_l2)

#err_H10_u = errornorm(u_DG1, u_ref, 'H10')#, degree_rise=0)
#print(err_H10_u)
#err_H10_phi = errornorm(phi_DG1, phi_ref, 'H10')#, degree_rise=0)
#print(err_H10_phi)
#tot_H10 = np.sqrt(err_H10_u**2+err_H10_phi**2)
err_H10 = errornorm(v_DG1, v_ref, 'H10')
print('Error H10: %.3e' % err_H10)

#Erreur H10 donne ce qu'il faut dans le cas de Cosserat ?
