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
    
# Mesh
h = 1e-3
nb_elt = 5
mesh = RectangleMesh(Point(-h/2,0),Point(h/2,h),nb_elt,5*nb_elt,"crossed")

# Parameters
nu = 0.3 # Poisson's ratio #Correct?
l = 1e-4 # intrinsic length scale
a = 2 #or 1???
G = 10e9 #Shear modulus
Gc = 20e9
M = G*l*l
E =  2*G*(1+nu) #Young Modulus

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

#boundaries
def top_down(x, on_boundary):
    return (near(x[1], 0) or near(x[1], h)) and on_boundary

# Sub domain for rotation at right end
def left_right(x, on_boundary):
    return (near(x[0], -h/2) or near(x[0], h/2)) and on_boundary

top_down_boundary = AutoSubDomain(top_down)
top_down_boundary.mark(boundary_parts, 1)
left_right_boundary = AutoSubDomain(left_right)
left_right_boundary.mark(boundary_parts, 2)
ds = ds(subdomain_data=boundary_parts)

#BC
u_D = Expression(('1e-5*x[1]/h','0'), h=h, degree=1)
phi_D = Expression('-0.1*x[1]/h', h=h, degree=1)

#ref
x = SpatialCoordinate(mesh)
K1 = 0 #TBD
K2 = 0
K3 = 0
tau_c = 0 #TBD
delta = 2*sqrt(G*Gc/((G+Gc)*M))
u_0 = -2*Gc/(G+Gc)*(K1/delta*exp(delta*x[1]) - K2/delta*exp(-delta*x[1])) + tau_c/G*x[1] + K3
omega = K1*exp(delta*x[1]) - K2*exp(-delta*x[1]) - 0.5*tau_c/G

#compliance tensor
problem.micropolar_constants(E, nu, l/2, a)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
inner_pen = inner_penalty(problem) #test
lhs += inner_pen

#Listing Dirichlet BC
bc = [[0,u_D[0],1], [1, u_D[1],1], [2, phi_D,1], [1, Constant(0), 2]]

#Nitsche penalty rhs
nitsche_and_bnd = rhs_bnd_penalty(problem, boundary_parts, bc) 
Rhs = nitsche_and_bnd

#Nitsche penalty bilinear form
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
lhs += bnd

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(PETScMatrix(lhs), v_DG.vector(), PETScVector(Rhs), 'mumps')
u_DG,phi_DG = v_DG.split()

#Computing reconstruction
v_DG1 = Function(problem.V_DG1)
v_DG1.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
u_DG1,phi_DG1 = v_DG1.split()

#plot
fig = plot(u_DG1[0])
plt.colorbar(fig)
plt.show()
fig = plot(u_DG1[1])
plt.colorbar(fig)
plt.show()
fig = plot(phi_DG1)
plt.colorbar(fig)
plt.show()
sys.exit()


#Computing stresses
strains = problem.strains_2d(u_DG1, phi_DG1)
sig,mu = problem.stresses_2d(strains)
deff,kappa = strains
