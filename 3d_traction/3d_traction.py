#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve,cg

# Parameters
nu = 0.3 #0.49 #0.3 # Poisson's ratio
mu = 1000.0 # shear modulus G
lmbda = ( 2.*mu*nu ) / (1-2*nu) # 1st Lame constant

l = 0.2 # intrinsic length scale
N = 0.93 # coupling parameter
    
# Mesh
L = 5
H = 1
nb_elt = 2
mesh = BoxMesh(Point(0., 0., 0.), Point(L, H, H), 5*nb_elt, nb_elt, nb_elt)

#Creating the DEM problem
problem = DEMProblem(mesh, 4*mu, 4*mu*l*l) #sure about second penalty term?
print('nb dofs: %i' % problem.nb_dof_DEM)

boundary_parts = MeshFunction("size_t", mesh, 1)
boundary_parts.set_all(0)
left = CompiledSubDomain("near(x[0], 0, 1e-4)")
right = CompiledSubDomain("near(x[0], %s, 1e-4)"%L)
left.mark(boundary_parts, 1) # mark left as 1
right.mark(boundary_parts, 2) # mark right as 2

u_D = Constant(2)
phi_D = Constant(0)

#Computing coefficients for Cosserat material
problem.micropolar_constants(nu, mu, lmbda, l, N)

# Variational problem
lhs = problem.elastic_bilinear_form()

#Penalty matrix
lhs += inner_penalty_light(problem)

#Listing Dirichlet BC
bc = [[0, Constant(0), 1], [0, u_D, 2], [0, Constant(0), 0]]
#Add bc to make problem isostatic!

#Nitsche penalty rhs
rhs = rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
lhs += lhs_bnd_penalty(problem, boundary_parts, bc)

#Solving linear problem
v = spsolve(lhs,rhs)
#v,info = cg(lhs,rhs)
#assert info == 0
print(v) #changing because reconstruction changes?
v_h = Function(problem.V_DG1)
#v_h = Function(problem.V_DG)
#v_h = interpolate(Constant((1,1,1,1,1,1)), problem.V_DG)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#U = VectorFunctionSpace(problem.mesh, 'DG', 1)
#u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 1)
#phi = interpolate(phi_D, U)

aux = project((u_h[0]-float(u_D))/float(u_D), U)
print(u_h(L,0,0)[0])
print(abs(aux(L,0,0)) * 100)

#print('norms')


file = File('3d_traction.pvd')

file << u_h
file << phi_h
U = TensorFunctionSpace(problem.mesh, 'DG', 0)
file << project(problem.strain_3d(u_h, phi_h), U)
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
fig = plot(u_h[2])
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
