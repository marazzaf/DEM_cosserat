#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from petsc4py import PETSc
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr
    
# Mesh
L = 0.5
nb_elt = 25 #40 # 80 #110
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

# Parameters
nu = 0.3 # Poisson's ratio
E = 1 #Young Modulus
l = L/100 # intrinsic length scale
a = 0.5

#Creating the DEM problem
cte = 2e2 #2e2
problem = DEMProblem(mesh, cte) #1e3 semble bien

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

#BC
u_D = Expression('1e-3*(x[0]+0.5*x[1])', degree=1)
v_D = Expression('1e-3*(x[0]+x[1])', degree=1)
alpha = -2 #-2
phi_D = Constant(0.25e-3*(1+alpha))
bc = [[0, u_D], [1, v_D], [2, phi_D]]

#compliance tensor
problem.micropolar_constants(E, nu, l, a)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
inner_pen = inner_penalty(problem) #test
lhs += inner_pen

#rhs
t = Expression(('-G*(2*(1-nu)/(1-2*nu)+1+a)','-G*(2*(1-nu)/(1-2*nu)+1+a)','2*a*(x[0]-x[1])*G'), G=problem.G, nu=nu, a=a, degree = 1)
Rhs = problem.assemble_volume_load(t)

#Listing Dirichlet BC
bc = [[0,u_D[0],0], [1, u_D[1],0], [2, phi_D,0]]

#Nitsche penalty rhs
nitsche_and_bnd = rhs_bnd_penalty(problem, boundary_parts, bc) 
Rhs += nitsche_and_bnd

#Nitsche penalty bilinear form
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
lhs += bnd


Lhs = sp.csr_matrix(lhs.getValuesCSR()[::-1], shape = lhs.size)
Rhs = Rhs.getArray()

truc = lsmr(Lhs, Rhs)
print('Condition number: %.1f' % truc[6])
