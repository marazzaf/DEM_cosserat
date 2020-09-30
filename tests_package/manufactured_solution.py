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
    gamma = as_vector([v[0].dx(0), v[1].dx(1), v[1].dx(0) - eta, v[0].dx(1) + eta])
    kappa = grad(eta)
    return gamma, kappa

def stress(D,strains):
    gamma,kappa = strains
    sigma = dot(D, gamma)
    mu = 4*G*l*l * kappa
    return sigma, mu
    
# Mesh
L = 0.5
nb_elt = 3
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l) #sure about second penalty term?

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

#ds = Measure('ds')(subdomain_data=boundary_parts)

x = SpatialCoordinate(problem.mesh)
u_1 = Expression('0.5*(x[0]*x[0]+x[1]*x[1])', degree=2)
#u_0 = Expression(('0.5*(x[0]*x[0]+x[1]*x[1])','0.5*(x[0]*x[0]+x[1]*x[1])','0'), degree=2)
phi = Constant(0)
bc = [[[0,1,2], [u_1,u_1,phi]]]

#compliance tensor
D = D_Matrix(G, nu, l, N)

# Variational problem
A = elastic_bilinear_form(problem, D, strain, stress)

#rhs

t = Constant((-(a+c),-(a+c),0))
rhs = assemble_volume_load(t, problem)

##Imposing weakly the BC!
#rhs += rhs_nitsche_penalty(problem, bc, D, strain, stress)
#
##Nitsche penalty bilinear form
#A += lhs_nitsche_penalty(problem, bc)
consistency = inner(dot(avg(sigma(du,alpha)),n('+')), jump(v))*dS + inner(jump(sigma(du,alpha),n), avg(v))*dS + inner(dot(sigma(du,alpha),n), v)*ds(2)
pen_bnd = eta/h * inner(du,v) * (ds(2) + ds(1))  - eta/h * inner(u_R,v) * ds(2) - inner(dot(sigma(du,alpha),n),u_R) * ds(2)


#Penalty matrix
#A += inner_penalty(problem)
A += inner_penalty_bis(problem)

#Solving linear problem
v = spsolve(A,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, psi_h = v_h.split()

fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_15.pdf')
plt.show()

U = FunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_1, U)
fig = plot(u)
plt.colorbar(fig)
plt.savefig('ref_u_x_15.pdf')
plt.show()
sys.exit()

fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_15.pdf')
plt.show()
fig = plot(psi_h)
plt.colorbar(fig)
plt.savefig('phi_15.pdf')
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
