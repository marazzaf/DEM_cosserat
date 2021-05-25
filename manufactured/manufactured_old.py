#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
    
# Mesh
L = 0.5
nb_elt = 40
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

# Parameters
nu = 0.3 # Poisson's ratio
E = 1 #Young Modulus
l = L # intrinsic length scale

#Creating the DEM problem
cte = 10 #1e2
problem = DEMProblem(mesh, cte) #1e3 semble bien

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

A = 0.5 #What value to put?
B = 1 #Same question
u_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])'), A=A, degree=2)
phi_D = Expression('B*(x[0]-x[1])', B=B, degree=1)
tot_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])', 'B*(x[0]-x[1])'), A=A, B=B, degree=2)

#compliance tensor
problem.micropolar_constants(E, nu, l)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
#inner_pen = inner_penalty_light(problem) #usual
inner_pen = inner_penalty(problem) #test
lhs += inner_pen
#lhs += inner_consistency(problem)

#rhs
#t = Expression(('-G*(2*A*(a+c)+B*(d-c))','-G*(2*A*(a+c)+B*(d-c))','-2*(x[0]-x[1] )*(d-c)*(B-A)*G'), G=G, A=A, B=B, a=a, b=b, c=c, d=d, degree = 1)
#lambda+4G ???
t = Expression(('-(lamda+2*G)','-(lamda+2*G)','2*(x[0]-x[1])*G'), G=problem.G, A=A, B=B, lamda=problem.lamda, degree = 1)
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

##plot
#fig = plot(u_DG1[0])
#plt.colorbar(fig)
#plt.show()
#fig = plot(u_DG1[1])
#plt.colorbar(fig)
#plt.show()
#fig = plot(phi_DG1)
#plt.colorbar(fig)
#plt.show()
#sys.exit()

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

#DG1 errors
err_grad = np.sqrt(errornorm(u_DG1, u, 'H10')**2 + errornorm(phi_DG1, phi, 'H10')**2)
#err_L2 = np.sqrt(errornorm(u_DG1, u, 'L2')**2 + errornorm(phi_DG1, phi, 'L2')**2)
err_L2 = errornorm(v_DG, v, 'L2')
print(problem.nb_dof_DEM)
print('L2 error: %.2e' % err_L2)
print('H10 error: %.2e' % err_grad)
#sys.exit()

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


