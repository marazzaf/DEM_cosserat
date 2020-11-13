#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve,cg,bicgstab

# Parameters
nu = 0.3 # Poisson's ratio
l = 0.2 # intrinsic length scale
N = 0.8 # coupling parameter
G = 100

#Parameters for D_Matrix
a = 2*(1-nu)/(1-2*nu)
b = 2*nu/(1-2*nu)
c = 1/(1-N*N)
d = (1-2*N*N)/(1-N*N)
    
# Mesh
L = 0.5
nb_elt = 80
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
#problem = DEMProblem(mesh, 2*G, 2*G*l*l) #sure about second penalty term?
problem = DEMProblem(mesh, 4*G, 4*G*l*l)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

A = 0.5 #What value to put?
B = 1 #Same question
u_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])'), A=A, degree=2)
phi_D = Expression('B*(x[0]-x[1])', B=B, degree=1)

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N, l)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

U = VectorFunctionSpace(problem.mesh, 'DG', 0)
u_DG0 = interpolate(u_D, U)
#u_DG0 = project(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 0)
phi_DG0 = interpolate(phi_D, U)
#phi_DG0 = project(phi_D, U)
sol_DG0 = project(as_vector((u_DG0[0],u_DG0[1],phi_DG0)), problem.V_DG).vector().get_local()

#ref solution CG 2
U = VectorFunctionSpace(problem.mesh, 'CG', 2)
ref_u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'CG', 1)
ref_phi = interpolate(phi_D, U)

#DG0 errors
err_L2 = np.sqrt(errornorm(u_DG0, ref_u, 'L2')**2 + errornorm(phi_DG0, ref_phi, 'L2')**2)
err_L2_u = errornorm(u_DG0, ref_u, 'L2')
err_L2_phi = errornorm(phi_DG0, ref_phi, 'L2')
print(err_L2)
print(err_L2_u)
print(err_L2_phi)


#DG1 errors
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * sol_DG0)
u_DG1,phi_DG1 = v_h.split()
err_grad = np.sqrt(errornorm(u_DG1, ref_u, 'H10')**2 + errornorm(phi_DG1, ref_phi, 'H10')**2)
print(err_grad)
sys.exit()

#DG0 L2 error
v_h = Function(problem.V_DG)
v_h.vector().set_local(v)
u_h, phi_h = v_h.split()
U = VectorFunctionSpace(problem.mesh, 'DG', 0)
ref_u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 0)
ref_phi = interpolate(phi_D, U)
err_L2 = np.sqrt(errornorm(u_h, ref_u, 'L2')**2 + errornorm(phi_h, ref_phi, 'L2')**2)
print(err_L2)

#print('Errors in energy:')
#print(err_energy)
#print('Tot: %.2f' % (np.sqrt(0.5 * np.dot(error, lhs*error))))
#print('No bnd energy: %.2f' % (np.sqrt(0.5 * np.dot(error, (inner+elas)*error))))
#print('Pen: %.2f' % (np.sqrt(0.5 * np.dot(error, (inner+bnd)*error))))
#print('Elas: %.2f' % (np.sqrt(0.5 * np.dot(error, elas*error))))
#print('Bnd pen: %.2f' % (np.sqrt(0.5 * np.dot(error, bnd*error))))
#print('Bnd pen squared: %.2f' % (np.dot(error, bnd*error)))

#print(errornorm(u_h, u, 'L2'))
#print(errornorm(phi_h, phi, 'L2'))

#For energy error
Mat = energy_error_matrix(problem, boundary_parts)

#Energy error
error = v - project(as_vector((ref_u[0],ref_u[1],ref_phi)), problem.V_DG).vector().get_local()
en_error = np.dot(error, Mat*error)
print('Error in energy norm: %.5e' % (np.sqrt(en_error)))

