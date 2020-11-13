#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve,cg,bicgstab

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
nb_elt = 80
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
#problem = DEMProblem(mesh, 2*G, 2*G*l*l) #sure about second penalty term?
problem = DEMProblem(mesh, 4*G, 4*G*l*l)

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
inner_pen = inner_penalty_light(problem) #light
lhs += inner_pen

#rhs
t = Expression(('-G*(2*A*(a+c)+B*(d-c))','-G*(2*A*(a+c)+B*(d-c))','-2*(x[0]-x[1] )*(d-c)*(B-A)*G'), G=G, A=A, B=B, a=a, b=b, c=c, d=d, degree = 1)
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
#v,info = bicgstab(lhs,rhs) #, tol=1e-10)
#assert info == 0
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


#fig = plot(u_h[0])
#plt.colorbar(fig)
###plt.savefig('u_x_25.pdf')
#plt.show()
#fig = plot(u[0])
#plt.colorbar(fig)
###plt.savefig('ref_u_x_25.pdf')
#plt.show()
##fig = plot(u_h[0]-u[0])
##plt.colorbar(fig)
##plt.show()
##
#fig = plot(u_h[1])
#plt.colorbar(fig)
###plt.savefig('u_y_25.pdf')
#plt.show()
#fig = plot(u[1])
#plt.colorbar(fig)
###plt.savefig('ref_u_y_25.pdf')
#plt.show()
##fig = plot(u_h[1]-u[1])
##plt.colorbar(fig)
##plt.show()
##
#fig = plot(phi_h)
#plt.colorbar(fig)
###plt.savefig('phi_25.pdf')
#plt.show()
#fig = plot(phi)
#plt.colorbar(fig)
###plt.savefig('ref_phi_25.pdf')
#plt.show()
##fig = plot(phi_h-phi)
##plt.colorbar(fig)
##plt.show()
##sys.exit()

#DG1 errors
err_grad = np.sqrt(errornorm(u_h, u, 'H10')**2 + errornorm(phi_h, phi, 'H10')**2)
err_L2 = np.sqrt(errornorm(u_h, u, 'L2')**2 + errornorm(phi_h, phi, 'L2')**2)
print(problem.nb_dof_DEM)
print(err_grad)
print(err_L2)

#DG0 L2 error
v_h = Function(problem.V_DG)
v_h.vector().set_local(v)
u_h, phi_h = v_h.split()
U = VectorFunctionSpace(problem.mesh, 'DG', 0)
ref_u = interpolate(u_D, U)
U = FunctionSpace(problem.mesh, 'DG', 0)
ref_phi = interpolate(phi_D, U)
err_L2 = np.sqrt(errornorm(u_h, ref_u, 'L2')**2 + errornorm(phi_h, ref_phi, 'L2')**2)
print(err_L2)

#print('Errors in energy:')
#print(err_energy)
#print('Tot: %.2f' % (np.sqrt(0.5 * np.dot(error, lhs*error))))
#print('No bnd energy: %.2f' % (np.sqrt(0.5 * np.dot(error, (inner+elas)*error))))
#print('Pen: %.2f' % (np.sqrt(0.5 * np.dot(error, (inner+bnd)*error))))
#print('Elas: %.2f' % (np.sqrt(0.5 * np.dot(error, elas*error))))
#print('Bnd pen: %.2f' % (np.sqrt(0.5 * np.dot(error, bnd*error))))
#print('Bnd pen squared: %.2f' % (np.dot(error, bnd*error)))

#print(errornorm(u_h, u, 'L2'))
#print(errornorm(phi_h, phi, 'L2'))

#For energy error
Mat = energy_error_matrix(problem, boundary_parts)

#Energy error
error = v - project(as_vector((ref_u[0],ref_u[1],ref_phi)), problem.V_DG).vector().get_local()
en_error = np.dot(error, Mat*error)
print('Error in energy norm: %.5e' % (np.sqrt(en_error)))

