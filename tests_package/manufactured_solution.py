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

def strain(v, eta):
    gamma = as_vector([v[0].dx(0), v[1].dx(0) - eta, v[0].dx(1) + eta, v[1].dx(1)])
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

#ds = Measure('ds')(subdomain_data=boundary_parts)

x = SpatialCoordinate(problem.mesh)
u_D = Expression(('0.5*(x[0]*x[0]+x[1]*x[1])','0.5*(x[0]*x[0]+x[1]*x[1])'), degree=2) #A=0.5
phi_D = Expression('x[0]-x[1]', degree=1) #B=1

#compliance tensor
problem.D = problem.D_Matrix(G, nu, l, N)

# Variational problem
A = elastic_bilinear_form(problem, strain, stresses)

#Penalty matrix
A += inner_penalty(problem)

#rhs
t = Expression(('-0.5*(2*(a+d) - b - c) - (c-d)', '-0.5*(2*(a+d) - b - c) - (c-d)', '(x[0]-x[1])*(c-d)*(1.5-2)'), a=a, b=b, c=c, d=d, degree=1)
#t = Constant((0,0,0))
rhs = problem.assemble_volume_load(t)

#Listing Dirichlet BC
bc = [[0,u_D[0]], [1, u_D[1]], [2, phi_D]]
#bc = [[0,Constant(0)], [1, Constant(0)], [2, Constant(0)]]

#Nitsche penalty rhs
rhs += rhs_nitsche_penalty(problem, strain, stresses, bc)

#Nitsche penalty bilinear form
A += lhs_nitsche_penalty(problem, strain, stresses, bc)

#Solving linear problem
v = spsolve(A,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#print(u_h(0,L),phi_h(0,L))
#print(u_h(0,0),phi_h(0,0))

#U = VectorFunctionSpace(problem.mesh, 'DG', 1)
#u = interpolate(u_D, U)
#print(u(0,0))
#sys.exit()

#fig = plot(u_h[0])
#plt.colorbar(fig)
#plt.savefig('u_x_15.pdf')
#plt.show()
#fig = plot(u_h[1])
#plt.colorbar(fig)
#plt.savefig('u_y_15.pdf')
#plt.show()
#fig = plot(phi_h)
#plt.colorbar(fig)
#plt.savefig('phi_15.pdf')
#plt.show()
#sys.exit()

fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_25.pdf')
plt.show()
#sys.exit()

U = VectorFunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_D, U)[0]
fig = plot(u)
plt.colorbar(fig)
plt.savefig('ref_u_x_25.pdf')
plt.show()
#sys.exit()

fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_25.pdf')
plt.show()
#sys.exit()

U = VectorFunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_D, U)[1]
fig = plot(u)
plt.colorbar(fig)
plt.savefig('ref_u_y_25.pdf')
plt.show()
#sys.exit()

fig = plot(phi_h)
plt.colorbar(fig)
plt.savefig('phi_25.pdf')
plt.show()

U = FunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(phi_D, U)
fig = plot(u)
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
