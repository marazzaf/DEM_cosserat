#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
import numpy as np
from DEM_cosserat.miscellaneous import *

# Mesh
L = 0.12
nb_elt = 25 #10 #25 #50

# Parameters
nu = 0.25 # Poisson's ratio
G = 1e3 #Second lamé coefficient
E = 2*(1+nu)*G #Young Modulus
l = 0.1 # intrinsic length scale
a = 0.5 #last param in law
pen = 1 #penalty parameter

mesh = RectangleMesh(Point(-L,-L/2),Point(L,L/2),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, pen)
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

#compliance tensor
problem.micropolar_constants(E, nu, l, a)

# Variational problem
A = problem.elastic_bilinear_form()

#Penalty matrix
A += inner_penalty(problem)

#BC
u_D = Expression('1e-3*(x[0]+0.5*x[1])', degree=1)
v_D = Expression('1e-3*(x[0]+x[1])', degree=1)
phi_D = Constant(0.25e-3)
bc = [[0, u_D], [1, v_D], [2, phi_D]]
#Assembling
A += lhs_bnd_penalty(problem, boundary_parts, bc)
rhs = rhs_bnd_penalty(problem, boundary_parts, bc)

#Solving linear problem
v_DG = Function(problem.V_DG)
solve(PETScMatrix(A), v_DG.vector(), PETScVector(rhs), 'mumps')
v_h = Function(problem.V_DG1)
v_h.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
u_h, phi_h = v_h.split()

#Computing stresses
strains = problem.strains_2d(u_h, phi_h)
sigma,mu = problem.stresses_2d(strains)

#Test
W = FunctionSpace(problem.mesh, 'DG', 0)
#Testing on all elements
#Testing stresses
sigma_00 = local_project(sigma[0,0], W).vector().get_local()
#assert (np.round(sigma_00, 4) == 4).all()
print('Min value: %.2e  Max value: %.2e   Error: %.2e' % (sigma_00.min(), sigma_00.max(), max((sigma_00.max() - 4) / 4 * 100, (sigma_00.min() - 4) / 4 * 100)))
sigma_11 = local_project(sigma[1,1], W).vector().get_local()
#assert (np.round(sigma_11, 4) == 4).all()
print('Min value: %.2e  Max value: %.2e   Error: %.2e' % (sigma_11.min(), sigma_11.max(), max((sigma_11.max() - 4) / 4 * 100, (sigma_11.min() - 4) / 4 * 100)))
sigma_01 = local_project(sigma[0,1], W).vector().get_local()
#assert (np.round(sigma_01, 4) == 1.5).all()
print('Min value: %.2e  Max value: %.2e   Error: %.2e' % (sigma_01.min(), sigma_01.max(), max((sigma_01.max() - 1.5) / 1.5 * 100, (sigma_01.min() - 1.5) / 1.5 * 100)))
sigma_10 = local_project(sigma[1,0], W).vector().get_local()
#assert (np.round(sigma_10, 4) == 1.5).all()
print('Min value: %.2e  Max value: %.2e   Error: %.2e' % (sigma_10.min(), sigma_10.max(), max((sigma_10.max() - 1.5) / 1.5 * 100, (sigma_10.min() - 1.5) / 1.5 * 100)))
#Testing moments
mu_0 = local_project(mu[0], W).vector().get_local()
#assert (np.round(mu_0, 4)  == 0).all()
print('Min value: %.2e  Max value: %.2e' % (mu_0.min(), mu_0.max()))
mu_1 = local_project(mu[1], W).vector().get_local()
#assert (np.round(mu_1, 4)  == 0).all()
print('Min value: %.2e  Max value: %.2e' % (mu_1.min(), mu_1.max()))

