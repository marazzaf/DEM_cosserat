#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *

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
nb_elt = 10
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
cte = 1e2
problem = DEMProblem(mesh, cte*G, cte*G, cte) #1e3 semble bien
#problem = DEMProblem(mesh, 8*G, 8*G*l*l)
#print('nb_dof: %i' % problem.nb_dof_DEM)
#print(mesh.hmax())
#sys.exit()

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

A = 0.5 #What value to put?
B = 1 #Same question
u_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])'), A=A, degree=2)
phi_D = Expression('B*(x[0]-x[1])', B=B, degree=1)
tot_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])', 'B*(x[0]-x[1])'), A=A, B=B, degree=2)

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N, l)
#characteristic_length = np.sqrt(4*l*l / max(a,b,c,d))

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
#inner_pen = inner_penalty_light(problem) #usual
inner_pen = inner_penalty(problem) #test
lhs += inner_pen
#lhs += inner_consistency(problem)

#rhs
t = Expression(('-G*(2*A*(a+c)+B*(d-c))','-G*(2*A*(a+c)+B*(d-c))','-2*(x[0]-x[1] )*(d-c)*(B-A)*G'), G=G, A=A, B=B, a=a, b=b, c=c, d=d, degree = 1)
#t = Constant((0, 0, 0)) #test
rhs_load = problem.assemble_volume_load(t)
rhs = np.zeros_like(rhs_load)
rhs += rhs_load

#Listing Dirichlet BC
bc = [[0,u_D[0],0], [1, u_D[1],0], [2, phi_D,0]]

#Nitsche penalty rhs
#nitsche_and_bnd = rhs_bnd_penalty_test(problem, u_D, phi_D)
nitsche_and_bnd = rhs_bnd_penalty(problem, boundary_parts, bc) 
rhs += nitsche_and_bnd

#Nitsche penalty bilinear form
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
#bnd = lhs_bnd_penalty_test(problem)
lhs += bnd

#Converting matrix
from petsc4py import PETSc
lhs = lhs.tocsr()
petsc_mat = PETSc.Mat().createAIJ(size=lhs.shape, csr=(lhs.indptr, lhs.indices,lhs.data))
lhs = PETScMatrix(petsc_mat)
bb = Function(problem.V_DG)
bb.vector().set_local(rhs)
rhs = bb.vector()

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(lhs, v_DG.vector(), rhs, 'mumps')
u_DG,phi_DG = v_DG.split()
vec_DG = v_DG.vector().get_local()

#Computing reconstruction
v_DG1 = Function(problem.V_DG1)
v_DG1.vector().set_local(problem.DEM_to_DG1 * vec_DG)
u_DG1,phi_DG1 = v_DG1.split()

##Bilan d'énergie
#elastic_energy = 0.5*np.dot(vec_DG, elas * vec_DG)
#inner_penalty_energy = 0.5*np.dot(vec_DG, inner_pen * vec_DG)
#bnd_penalty_energy = 0.5*np.dot(vec_DG, bnd * vec_DG)
#work = np.dot(nitsche_and_bnd, vec_DG)
#work_load = np.dot(rhs_load, vec_DG)
#print('Elastic: %.2e' % elastic_energy)
#print('Inner pen: %.2e' % inner_penalty_energy)
#print('Bnd pen: %.2e' % bnd_penalty_energy)
#print('Work rhs bnd pen: %.2e' % work)
#print('Work volume load: %.2e' % work_load)

#solution de ref
U = VectorFunctionSpace(problem.mesh, 'CG', 2)
u = interpolate(u_D, U)
S = FunctionSpace(problem.mesh, 'CG', 1)
phi = interpolate(phi_D, S)
U = VectorElement('CG', mesh.ufl_cell(), 2)
S = FiniteElement('CG', mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement(U,S))
v = project(as_vector((u[0],u[1],phi)), V)

#Paraview output
rotation = File('out_%i.pvd' % nb_elt)
rotation << phi_DG
rotation << phi_DG1
rotation << phi

#Errors
h = CellDiameter(problem.mesh)
h_avg = 0.5 * (h('+') + h('-'))
diff_u = u_DG1 - u
diff_phi = phi_DG1 - phi

#Bnd error
error_u_bnd = assemble(inner(diff_u, diff_u) / h * ds)
print('Error bnd u: %.2e' % (np.sqrt(error_u_bnd)))
error_phi_bnd = assemble(l*l*inner(diff_phi, diff_phi) / h * ds)
print('Error bnd phi: %.2e' % (np.sqrt(error_phi_bnd)))

#Inner error
error_u_inner = assemble(inner(jump(diff_u), jump(diff_u)) / h_avg * dS)
print('Error inner u: %.2e' % (np.sqrt(error_u_inner)))
error_phi_inner = assemble(l*l*inner(jump(diff_phi), jump(diff_phi)) / h_avg * dS)
print('Error inner phi: %.2e' % (np.sqrt(error_phi_inner)))

#Grad error
error_u_grad = assemble(inner(grad(diff_u),grad(diff_u)) * dx)
print('Error grad u: %.2e' % (np.sqrt(error_u_grad)))
error_phi_grad = assemble(inner(grad(diff_phi),grad(diff_phi)) * dx)
print('Error grad phi: %.2e' % (np.sqrt(error_phi_grad)))

### Stress
#eps,kappa = problem.strains(u_h, phi_h)
##sigma_yy = project(sigma[1])
#file << project(kappa, U)
#sys.exit()

##test BC
#fig = plot(abs(u_DG[0]-u[0]))
#plt.colorbar(fig)
#plt.show()
#fig = plot(abs(u_DG[1]-u[1]))
#plt.colorbar(fig)
#plt.show()
#fig = plot(abs(phi_DG-phi))
#plt.colorbar(fig)
#plt.show()
#sys.exit()


#fig = plot(u_DG1[0])
#plt.colorbar(fig)
##plt.savefig('u_x_80.pdf')
#plt.show()
####fig = plot(u[0])
####plt.colorbar(fig)
######plt.savefig('ref_u_x_25.pdf')
####plt.show()
###fig = plot(u_DG1[0]-u[0])
###plt.colorbar(fig)
###plt.show()
###
#fig = plot(u_DG1[1])
#plt.colorbar(fig)
##plt.savefig('u_y_80.pdf')
#plt.show()
####fig = plot(u[1])
####plt.colorbar(fig)
######plt.savefig('ref_u_y_25.pdf')
####plt.show()
###fig = plot(u_DG1[1]-u[1])
###plt.colorbar(fig)
###plt.show()
####
#fig = plot(phi_DG1)
#plt.colorbar(fig)
##plt.savefig('phi_80.pdf')
#plt.show()
####fig = plot(phi)
####plt.colorbar(fig)
######plt.savefig('ref_phi_25.pdf')
####plt.show()
###fig = plot(phi_DG1-phi)
###plt.colorbar(fig)
###plt.show()
###sys.exit()

#DG1 errors
err_grad = np.sqrt(errornorm(u_DG1, u, 'H10')**2 + errornorm(phi_DG1, phi, 'H10')**2)
#err_L2 = np.sqrt(errornorm(u_DG1, u, 'L2')**2 + errornorm(phi_DG1, phi, 'L2')**2)
err_L2 = errornorm(v_DG, v, 'L2')
print(problem.nb_dof_DEM)
print('L2 error: %.2e' % err_L2)
print('H10 error: %.2e' % err_grad)
#sys.exit()

##DG0 L2 error
#v_h = Function(problem.V_DG)
#v_h.vector().set_local(v)
#u_h, phi_h = v_h.split()
##U = VectorFunctionSpace(problem.mesh, 'DG', 0)
##ref_u = interpolate(u_D, U)
##U = FunctionSpace(problem.mesh, 'DG', 0)
##ref_phi = interpolate(phi_D, U)
###err_L2 = np.sqrt(errornorm(u_h, ref_u, 'L2')**2 + errornorm(phi_h, ref_phi, 'L2')**2)
#err_L2 = np.sqrt(errornorm(u_h, u, 'L2')**2 + errornorm(phi_h, phi, 'L2')**2)
#print(err_L2)

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

##For energy error
#Mat = energy_error_matrix(problem, boundary_parts)
#
##Energy error
#ref_u,ref_phi,ref = DEM_interpolation(tot_D, problem)
##project(as_vector((ref_u[0],ref_u[1],ref_phi)), problem.V_DG).vector().get_local()
#error = v - ref
#en_error = np.dot(error, Mat*error)
#print('Error in energy norm: %.5e' % (np.sqrt(en_error)))

#error bnd
h = CellDiameter(problem.mesh)
h_avg = 0.5 * (h('+') + h('-'))
n = FacetNormal(problem.mesh)
diff_u = u_DG1 - u
error_u = assemble(inner(diff_u, diff_u) / h * ds + inner(jump(diff_u), jump(diff_u)) / h_avg * dS + inner(grad(diff_u),grad(diff_u)) * dx)
#error_u = assemble(inner(diff_u, diff_u) / h * ds + h * inner(dot(grad(diff_u), n), dot(grad(diff_u), n)) * ds + inner(jump(diff_u), jump(diff_u)) / h_avg * dS + h_avg * inner(dot(avg(grad(diff_u)), n('+')), dot(avg(grad(diff_u)), n('+'))) * dS)
#error_u = assemble(inner(diff_u, diff_u) / h * ds + h * inner(dot(grad(diff_u), n), dot(grad(diff_u), n)) * ds + inner(jump(diff_u), jump(diff_u)) / h_avg * dS)
#error_u = assemble(inner(diff_u, diff_u) / h * ds + inner(jump(diff_u), jump(diff_u)) / h_avg * dS)

diff_phi = phi_DG1 - phi
error_phi = assemble(inner(diff_phi, diff_phi) / h * ds + inner(jump(diff_phi), jump(diff_phi)) / h_avg * dS + inner(grad(diff_phi),grad(diff_phi)) * dx)
#error_phi = assemble(l*l*inner(diff_phi, diff_phi) / h * ds + l*l*inner(jump(diff_phi), jump(diff_phi)) / h_avg * dS + inner(grad(diff_phi),grad(diff_phi)) * dx)
#error_phi = assemble(inner(diff_phi, diff_phi) / h * ds + h * inner(dot(grad(diff_phi), n), dot(grad(diff_phi), n)) * ds + inner(jump(diff_phi), jump(diff_phi)) / h_avg * dS + h_avg * inner(dot(avg(grad(diff_phi)), n('+')), dot(avg(grad(diff_phi)), n('+'))) * dS)
#error_phi = assemble(inner(diff_phi, diff_phi) / h * ds + h * inner(dot(grad(diff_phi), n), dot(grad(diff_phi), n)) * ds + inner(jump(diff_phi), jump(diff_phi)) / h_avg * dS)
#error_phi = assemble(inner(diff_phi, diff_phi) / h * ds + inner(jump(diff_phi), jump(diff_phi)) / h_avg * dS)
energy_error = np.sqrt(error_u + error_phi)
print('Error in energy norm: %.2e' % energy_error)
print('%i %.2e %.2e %.2e' % (problem.nb_dof_DEM, err_L2, err_grad, energy_error))

