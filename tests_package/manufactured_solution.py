#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve

# Parameters
d = 2 #2d problem
nu = 0.3 # Poisson's ratio
l = 0.2 # intrinsic length scale
N = 0.8 # coupling parameter
G = 1

#Parameters for D_Matrix
a = 2*(1-nu)/(1-2*nu)
b = 2*nu/(1-2*nu)
c = 1/(1-N*N)
d = (1-2*N*N)/(1-N*N)

def D_Matrix(G, nu, l, N):
    return as_matrix([[a,0,0,b], [b,0,0,a], [0,c,d,0], [0,d,c,0]])

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
nb_elt = 15
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l) #sure about second penalty term?

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

#ds = Measure('ds')(subdomain_data=boundary_parts)

x = SpatialCoordinate(problem.mesh)
u_D = Expression(('0.5*(x[0]*x[0]+x[1]*x[1])','0.5*(x[0]*x[0]+x[1]*x[1])'), degree=2)
phi_D = Constant(0.)

#compliance tensor
problem.D = D_Matrix(G, nu, l, N)

# Variational problem
A = elastic_bilinear_form(problem, strain, stresses)

#Penalty matrix
A += inner_penalty(problem)

#rhs
t = Constant((-(a+c),-(a+c),0))
#t = Constant((0,0,0))
rhs = problem.assemble_volume_load(t)

#Listing Dirichlet BC
#bc = [[0,u_D[0]], [1, u_D[1]], [2, phi_D]]
bc = [[0,Constant(0)], [1, Constant(0)], [2, Constant(0)]]

#Nitsche penalty rhs
#rhs += rhs_nitsche_penalty(problem, strain, stresses, bc)

#Nitsche penalty bilinear form
A += lhs_nitsche_penalty(problem, strain, stresses, bc)

#Solving linear problem
v = spsolve(A,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#print(u_h(0,L),phi_h(0,L))
#U = VectorFunctionSpace(problem.mesh, 'DG', 1)
#u = interpolate(u_D, U)
#print(u(0,L))
#sys.exit()

fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_15.pdf')
plt.show()
fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_15.pdf')
plt.show()
fig = plot(phi_h)
plt.colorbar(fig)
plt.savefig('phi_15.pdf')
plt.show()
sys.exit()

U = VectorFunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_D, U)[1]
fig = plot(u)
plt.colorbar(fig)
plt.savefig('ref_u_y_15.pdf')
plt.show()
sys.exit()

fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_15.pdf')
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

error = abs((sigma_yy(10.0, 1e-6) - SCF) / SCF)

elements_size.append(hm)
SCF_0.append(sigma_yy(10.0, 1e-6))
errors.append(error)
        
print("Analytical SCF: %.5e" % SCF)
print(elements_size)
print(errors)
print(SCF_0)


file = File("sigma.pvd")
file << sigma_yy

plt.plot(elements_size, errors, "-*", linewidth=2)
plt.xlabel("elements size")
plt.ylabel("error")
plt.show()
