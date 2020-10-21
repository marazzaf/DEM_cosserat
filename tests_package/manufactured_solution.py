#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve

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

#def strain(v, eta):
#    gamma = as_vector([v[0].dx(0), v[1].dx(0) - eta, v[0].dx(1) + eta, v[1].dx(1)])
#    kappa = grad(eta)
#    return gamma, kappa

def strain(v, eta):
    gamma = as_vector([v[0].dx(0), v[0].dx(1) + eta, v[1].dx(0) - eta, v[1].dx(1)]) #correct
    kappa = grad(eta)
    return gamma, kappa

def stresses(D,strains):
    gamma,kappa = strains
    sigma = dot(D, gamma)
    mu = 4*G*l*l * kappa
    return sigma, mu
    
# Mesh
L = 0.5
nb_elt = 25
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l) #sure about second penalty term?

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

x = SpatialCoordinate(problem.mesh)
A = 10 #What value to put?
B = 12 #Same question
u_D = Expression(('A*(x[0]*x[0]+x[1]*x[1]-x[0]*x[1]-1)','A*(x[0]*x[0]+x[1]*x[1]-x[0]*x[1]-1)'), A=A, degree=3)
phi_D = Expression('B*(x[0]-x[1])', B=B, degree=2)

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N, l)

# Variational problem
lhs = problem.elastic_bilinear_form()

#Penalty matrix
lhs += inner_penalty(problem)

#rhs
t = Expression(('-(2*A*(a+d)+B*(c-d))','-(2*A*(a+d)+B*(c-d))','2*(x[0]-x[1])*(c-d)*(A-B)'), A=A, B=B, a=a, b=b, c=c, d=d, degree = 2)
rhs = problem.assemble_volume_load(t)

#Listing Dirichlet BC
bc = [[0,u_D[0]], [1, u_D[1]], [2, phi_D]]

#Nitsche penalty rhs
rhs += rhs_nitsche_penalty(problem, bc)

#Nitsche penalty bilinear form
lhs += lhs_nitsche_penalty(problem, bc)

#Solving linear problem
v = spsolve(lhs,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#print(u_h(0,L),phi_h(0,L))
#print(u_h(0,0),phi_h(0,0))

U = VectorFunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_D, U)
#print(u(0,0))
#sys.exit()

fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_25.pdf')
plt.show()
fig = plot(u[0])
plt.colorbar(fig)
plt.savefig('ref_u_x_25.pdf')
plt.show()

#fig = plot(u_h[1])
#plt.colorbar(fig)
#plt.savefig('u_y_25.pdf')
#plt.show()
#fig = plot(u[1])
#plt.colorbar(fig)
#plt.savefig('ref_u_y_25.pdf')
#plt.show()

fig = plot(phi_h)
plt.colorbar(fig)
plt.savefig('phi_25.pdf')
plt.show()

U = FunctionSpace(problem.mesh, 'DG', 1)
phi = interpolate(phi_D, U)
fig = plot(phi)
plt.colorbar(fig)
plt.savefig('ref_phi_25.pdf')
plt.show()
sys.exit()

# Stress
epsilon = strain(u_h, psi_h)
sigma = D*epsilon
sigma_yy = project(sigma[1])
#Other version
#epsilon = strain_bis(u_h, psi_h)
#sigma = stress(epsilon)[0]
#sigma_yy = project(sigma[1])


file = File("sigma.pvd")
file << sigma_yy
