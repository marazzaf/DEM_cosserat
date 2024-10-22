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

# Parameters
nu = 0.3 #No Poisson effect # Poisson's ratio
mu = 1000.0 # shear modulus G
lmbda = ( 2.*mu*nu ) / (1-2*nu) # 1st Lame constant

l = 0.2 # intrinsic length scale
N = 0.93 # coupling parameter
    
# Mesh
L = 5
H = 1
nb_elt = 5
mesh = BoxMesh(Point(0., 0., 0.), Point(L, H, H), 5*nb_elt, nb_elt, nb_elt)

#Creating the DEM problem
cte = 2
problem = DEMProblem(mesh, cte*mu, cte*mu*l*l) #sure about second penalty term?
print('nb dofs: %i' % problem.nb_dof_DEM)

boundary_parts = MeshFunction("size_t", mesh, problem.dim-1)
boundary_parts.set_all(0)
left = CompiledSubDomain("near(x[0], 0, 1e-4)")
right = CompiledSubDomain("near(x[0], %s, 1e-4)"%L)
left.mark(boundary_parts, 1) # mark left as 1
right.mark(boundary_parts, 2) # mark right as 2

u_D = Constant(2)
u_0 = Constant(0)

#Computing coefficients for Cosserat material
problem.micropolar_constants(nu, mu, lmbda, l, N)

# Variational problem
lhs = problem.elastic_bilinear_form()

#Penalty matrix
lhs += inner_penalty_light(problem)

#Listing Dirichlet BC
bc = [[0, u_0, 1], [0, u_D, 2], [1, u_0, 1], [2, u_0, 1], [3, u_0, 1], [4, u_0, 1], [5, u_0, 1]]
#Add bc to make problem isostatic!

#Nitsche penalty rhs
rhs = rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
lhs += lhs_bnd_penalty(problem, boundary_parts, bc)

#Solving linear problem
A = lhs.tocsr()
petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices,A.data))
A_aux = PETScMatrix(petsc_mat)
b = Function(problem.V_DG)
b.vector().set_local(rhs)
v_DG = Function(problem.V_DG)
print('Solve!')
solve(A_aux, v_DG.vector(), b.vector(), 'mumps')
u_DG, phi_DG = v_DG.split()
v_DG1 = Function(problem.V_DG1)
v_DG1.vector().set_local(problem.DEM_to_DG1 * v_DG.vector().get_local())
u_DG1, phi_DG1 = v_DG1.split()

#U = VectorFunctionSpace(problem.mesh, 'DG', 1)
#u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 1)
#phi = interpolate(phi_D, U)

aux = project((u_DG1[0]-float(u_D))/float(u_D), U)
print(u_DG1(L,0,0)[0])
print(abs(aux(L,0,0)) * 100)

#print('norms')


file = File('3d_traction.pvd')

file << u_DG
file << u_DG1
file << phi_DG
U = TensorFunctionSpace(problem.mesh, 'DG', 0)
file << project(problem.strain_3d(u_DG1, phi_DG1), U)
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
