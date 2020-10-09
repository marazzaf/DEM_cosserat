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
problem = DEMProblem(mesh, 2*G, 2*G*l*l) #sure about second penalty term? #2*G*l

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N)

# Variational problem
A = elastic_bilinear_form(problem, strain, stresses)

#Penalty matrix
A += inner_penalty(problem)

#rhs
t = Constant((-(a+b),-(a+b),0))
rhs = problem.assemble_volume_load(t)

#Nitsche penalty bilinear form. Homogeneous Dirichlet in this case.
A += lhs_nitsche_penalty(problem, strain, stresses)

#Solving linear problem
v = spsolve(A,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#assert abs(np.linalg.norm(u_h(0,L))) < abs(np.linalg.norm(u_h(0,0))) / 10
#assert abs(np.linalg.norm(phi_h(0,L))) < abs(np.linalg.norm(phi_h(0,0))) / 100
#print(u_h(0,L),phi_h(0,L))
print(u_h(0,0),phi_h(0,0))

#Reference solution
x = SpatialCoordinate(problem.mesh)
u_D = Expression(('B*((x[0]*x[0]+x[1]*x[1]-x[0]*x[1])/(L*L) - 1)','B*((x[0]*x[0]+x[1]*x[1]-x[0]*x[1])/(L*L) - 1)'), B=L**2/G,L=L,  degree=2)
phi_D = Constant(0.)

U = VectorFunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_D, U)
print(u(0,0))

gamma,kappa = strain(u_h,phi_h)

#U = FunctionSpace(problem.mesh, 'DG', 0)
#fig = plot(local_project(gamma[3], U))
#plt.colorbar(fig)
#plt.savefig('stress_1_1.pdf')
#plt.show()
#fig = plot(local_project(kappa[1],U))
#plt.colorbar(fig)
#plt.savefig('kappa_1.pdf')
#plt.show()
#sys.exit()

fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_25.pdf')
plt.show()
fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_25.pdf')
plt.show()
fig = plot(phi_h)
plt.colorbar(fig)
plt.savefig('phi_25.pdf')
plt.show()
sys.exit()
