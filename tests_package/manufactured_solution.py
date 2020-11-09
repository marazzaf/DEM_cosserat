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
L = 0.5
nb_elt = 20
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l*l) #sure about second penalty term?

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

A = 0.5 #What value to put?
B = 1 #Same question
u_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])'), A=A, degree=2)
phi_D = Expression('B*(x[0]-x[1])', B=B, degree=1)

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N, l)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
inner = inner_penalty_light(problem)
lhs += inner

#rhs
t = Expression(('-G*(2*A*(a+c)+B*(d-c))','-G*(2*A*(a+c)+B*(d-c))','-2*(x[0]-x[1])*(d-c)*(B-A)*G'), G=G, A=A, B=B, a=a, b=b, c=c, d=d, degree = 1)
#t = Constant((0, 0, 0)) #test
rhs = problem.assemble_volume_load(t)

#Listing Dirichlet BC
bc = [[0,u_D[0],0], [1, u_D[1],0], [2, phi_D,0]]

#Nitsche penalty rhs
#rhs += rhs_nitsche_penalty(problem, bc)
rhs += rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
#lhs += lhs_nitsche_penalty(problem, bc)
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
lhs += bnd

#Solving linear problem
v = spsolve(lhs,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#print(u_h(0,L),phi_h(0,L))
#print(u_h(0,0),phi_h(0,0))

U = VectorFunctionSpace(problem.mesh, 'DG', 1)
u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 1)
phi = interpolate(phi_D, U)

#file = File('out.pvd')
#
#file << u_h
#file << phi_h
#
### Stress
#eps,kappa = problem.strains(u_h, phi_h)
##sigma_yy = project(sigma[1])
#file << project(kappa, U)
#sys.exit()

##test BC
#fig = plot(abs(u_h[0]-u[0]))
#plt.colorbar(fig)
#plt.show()
#fig = plot(abs(u_h[1]-u[1]))
#plt.colorbar(fig)
#plt.show()
#fig = plot(abs(phi_h-phi))
#plt.colorbar(fig)
#plt.show()
#sys.exit()


fig = plot(u_h[0])
plt.colorbar(fig)
##plt.savefig('u_x_25.pdf')
plt.show()
fig = plot(u[0])
plt.colorbar(fig)
##plt.savefig('ref_u_x_25.pdf')
plt.show()
#fig = plot(u_h[0]-u[0])
#plt.colorbar(fig)
#plt.show()
#
fig = plot(u_h[1])
plt.colorbar(fig)
##plt.savefig('u_y_25.pdf')
plt.show()
fig = plot(u[1])
plt.colorbar(fig)
##plt.savefig('ref_u_y_25.pdf')
plt.show()
#fig = plot(u_h[1]-u[1])
#plt.colorbar(fig)
#plt.show()
#
fig = plot(phi_h)
plt.colorbar(fig)
##plt.savefig('phi_25.pdf')
plt.show()
fig = plot(phi)
plt.colorbar(fig)
##plt.savefig('ref_phi_25.pdf')
plt.show()
#fig = plot(phi_h-phi)
#plt.colorbar(fig)
#plt.show()
#sys.exit()

#write convergence test to see if okay...
err_grad = np.sqrt(errornorm(u_h, u, 'H10')**2 + errornorm(phi_h, phi, 'H10')**2)
err_energy = np.sqrt(0.5 * np.dot(v, bnd*v)) #lhs #elas #inner #bnd
err_L2 = np.sqrt(errornorm(u_h, u, 'L2')**2 + errornorm(phi_h, phi, 'L2')**2)
print(problem.nb_dof_DEM)
print(err_grad)
print(err_energy)
print(err_L2)
#print(errornorm(u_h, u, 'L2'))
#print(errornorm(phi_h, phi, 'L2'))

