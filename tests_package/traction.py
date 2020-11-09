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
    
# Mesh
L = 5
H = 1
nb_elt = 40
mesh = RectangleMesh(Point(0,0.5*H),Point(L,-0.5*H),5*nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l*l) #sure about second penalty term?

boundary_parts = MeshFunction("size_t", mesh, 1)
boundary_parts.set_all(0)
left = CompiledSubDomain("near(x[0], 0, 1e-4)")
right = CompiledSubDomain("near(x[0], %s, 1e-4)"%L)
left.mark(boundary_parts, 1) # mark left as 1
right.mark(boundary_parts, 2) # mark right as 2

u_D = Constant(2)
phi_D = Constant(0)

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N, l)

# Variational problem
lhs = problem.elastic_bilinear_form()

#Penalty matrix
lhs += inner_penalty(problem)

#Listing Dirichlet BC
bc = [[0, Constant(0), 1], [0, u_D, 2]] #[2, phi_D, 1]
#bc = [[0, Constant(0)], [0, u_D]]
#u_D = Expression(('2*x[0]/L', '0'), L=L, degree=1)
#bc = [[0, u_D]]

#Nitsche penalty rhs
#rhs += rhs_nitsche_penalty(problem, bc)
rhs = rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
#lhs += lhs_nitsche_penalty(problem, bc)
lhs += lhs_bnd_penalty(problem, boundary_parts, bc)

#Solving linear problem
v = spsolve(lhs,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#U = VectorFunctionSpace(problem.mesh, 'DG', 1)
#u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 1)
#phi = interpolate(phi_D, U)

aux = project((u_h[0]-float(u_D))/float(u_D), U)
print(u_h(L,0)[0])
print(abs(aux(L,0)) * 100)


file = File('traction.pvd')

file << u_h
file << phi_h
sys.exit()

fig = plot(u_h[0])
plt.colorbar(fig)
##plt.savefig('u_x_25.pdf')
plt.show()
#fig = plot(u[0])
#plt.colorbar(fig)
##plt.savefig('ref_u_x_25.pdf')
#plt.show()
#fig = plot(u_h[0]-u[0])
#plt.colorbar(fig)
#plt.show()
#
fig = plot(u_h[1])
plt.colorbar(fig)
##plt.savefig('u_y_25.pdf')
plt.show()
#fig = plot(u[1])
#plt.colorbar(fig)
##plt.savefig('ref_u_y_25.pdf')
#plt.show()
#fig = plot(u_h[1]-u[1])
#plt.colorbar(fig)
#plt.show()
#
fig = plot(phi_h)
plt.colorbar(fig)
##plt.savefig('phi_25.pdf')
plt.show()
#fig = plot(phi)
#plt.colorbar(fig)
##plt.savefig('ref_phi_25.pdf')
#plt.show()
#fig = plot(phi_h-phi)
#plt.colorbar(fig)
#plt.show()
sys.exit()

## Stress
#epsilon = problem.strain(u_h, psi_h)
#sigma = D*epsilon
#sigma_yy = project(sigma[1])

#write convergence test to see if okay...
err_grad = np.sqrt(errornorm(u_h, u, 'H10')**2 + errornorm(phi_h, phi, 'H10')**2)
err_energy = np.sqrt(np.dot(v, lhs*v))
err_L2 = np.sqrt(errornorm(u_h, u, 'L2')**2 + errornorm(phi_h, phi, 'L2')**2)
print(problem.nb_dof_DEM)
print(err_grad)
print(err_energy)
print(err_L2)
print(errornorm(u_h, u, 'L2'))
print(errornorm(phi_h, phi, 'L2'))

