#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
import numpy as np
from DEM_cosserat.miscellaneous import *
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

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
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

#Functionnal spaces
U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
print('nb dof CG: %i' % V.dofmap().global_dimension())
U,S = V.split()
U_1, U_2 = U.sub(0), U.sub(1)

def strain(v,psi):
    e = grad(v) + as_tensor(((0, 1), (-1, 0))) * psi
    kappa = grad(psi)
    return e,kappa

def stress(e, kappa):
    eps = as_vector((e[0,0], e[1,1], e[0,1], e[1,0]))
    aux_1 = 2*(1-nu)/(1-2*nu)
    aux_2 = 2*nu/(1-2*nu)
    Mat = G * as_tensor(((aux_1,aux_2,0,0), (aux_2, aux_1,0,0), (0,0,1+a,1-a), (0,0,1-a,1+a)))
    sig = dot(Mat, eps)
    sigma = as_tensor(((sig[0], sig[2]), (sig[3], sig[1])))
    mu = 4*G*l*l * kappa
    return sigma, mu

#BC
u_D = Expression('1e-3*(x[0]+0.5*x[1])', degree=1)
v_D = Expression('1e-3*(x[0]+x[1])', degree=1)
alpha = -2
phi_D = Constant(0.25e-3*(1+alpha))
bc_1 = DirichletBC(U_1, u_D, boundary_parts, 0)
bc_2 = DirichletBC(U_2, v_D, boundary_parts, 0)
bc_3 = DirichletBC(S, phi_D, boundary_parts, 0)
bcs = [bc_1, bc_2, bc_3]

# Variational problem
u, phi = TrialFunctions(V)
v, psi = TestFunctions(V)
e,kappa = strain(u,phi)
sigma,mu = stress(e,kappa)
e,kappa = strain(v,psi)
A = inner(sigma, e)*dx + inner(mu, kappa)*dx
#Volume load
q = Constant(1)
L = inner(q, psi) * dx

#computing condition number
Lhs = as_backend_type(assemble(A)).mat()
Lhs = sp.csr_matrix(Lhs.getValuesCSR()[::-1], shape = Lhs.size)
Rhs = as_backend_type(assemble(L)).vec()
Rhs = Rhs.getArray()
truc = lsmr(Lhs, Rhs)
print('Condition number: %.1f' % truc[6])

#Solving linear problem
sol = Function(V)
solve(A == L, sol, bcs=bcs)
u_h, phi_h = sol.split()

#Computing stresses
e_h, kappa_h = strain(u_h, phi_h)
sigma,mu = stress(e_h, kappa_h)

#Test
W = FunctionSpace(mesh, 'DG', 0)
#Testing stresses
sigma_00 = local_project(sigma[0,0], W).vector().get_local()
print('Min value: %.2e  Max value: %.2e   Error: %.2f%%' % (sigma_00.min(), sigma_00.max(), max((sigma_00.max() - 4) / 4 * 100, (sigma_00.min() - 4) / 4 * 100)))
sigma_11 = local_project(sigma[1,1], W).vector().get_local()
print('Min value: %.2e  Max value: %.2e   Error: %.2f%%' % (sigma_11.min(), sigma_11.max(), max((sigma_11.max() - 4) / 4 * 100, (sigma_11.min() - 4) / 4 * 100)))
sigma_01 = local_project(sigma[0,1], W).vector().get_local()
print('Min value: %.2e  Max value: %.2e   Error: %.2f%%' % (sigma_01.min(), sigma_01.max(), max((sigma_01.max() - 1) / 1 * 100, (sigma_01.min() - 1) / 1 * 100)))
sigma_10 = local_project(sigma[1,0], W).vector().get_local()
print('Min value: %.2e  Max value: %.2e   Error: %.2f%%' % (sigma_10.min(), sigma_10.max(), max((sigma_10.max() - 2) / 2 * 100, (sigma_10.min() - 2) / 2 * 100)))
#Testing moments
mu_0 = local_project(mu[0], W).vector().get_local()
print('Min value: %.2e  Max value: %.2e' % (mu_0.min(), mu_0.max()))
mu_1 = local_project(mu[1], W).vector().get_local()
print('Min value: %.2e  Max value: %.2e' % (mu_1.min(), mu_1.max()))

