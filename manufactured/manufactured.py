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
    
# Mesh
L = 0.5
nb_elt = 40 #40 # 80 #110
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

# Parameters
nu = 0.3 # Poisson's ratio
E = 1 #Young Modulus
l = L/100 # intrinsic length scale
a = 0.5

#Creating the DEM problem
cte = 1e2 #1e2
problem = DEMProblem(mesh, cte) #1e3 semble bien

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

u_D = Expression(('0.5*(x[0]*x[0]+x[1]*x[1])','0.5*(x[0]*x[0]+x[1]*x[1])'), degree=2)
phi_D = Expression('(x[0]-x[1])', degree=1)
tot_D = Expression(('0.5*(x[0]*x[0]+x[1]*x[1])','0.5*(x[0]*x[0]+x[1]*x[1])', '(x[0]-x[1])'), degree=2)

#compliance tensor
problem.micropolar_constants(E, nu, l, a)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
#inner_pen = inner_penalty_light(problem) #usual
inner_pen = inner_penalty(problem) #test
lhs += inner_pen
#lhs += inner_consistency(problem)

#rhs
t = Expression(('-G*(2*(1-nu)/(1-2*nu)+1+a)','-G*(2*(1-nu)/(1-2*nu)+1+a)','2*a*(x[0]-x[1])*G'), G=problem.G, nu=nu, a=a, degree = 1)
Rhs = problem.assemble_volume_load(t)

#Listing Dirichlet BC
bc = [[0,u_D[0],0], [1, u_D[1],0], [2, phi_D,0]]

#Nitsche penalty rhs
#nitsche_and_bnd = rhs_bnd_penalty_test(problem, u_D, phi_D)
nitsche_and_bnd = rhs_bnd_penalty(problem, boundary_parts, bc) 
Rhs += nitsche_and_bnd

#Nitsche penalty bilinear form
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
#bnd = lhs_bnd_penalty_test(problem)
lhs += bnd

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(PETScMatrix(lhs), v_DG.vector(), PETScVector(Rhs), 'mumps')
u_DG,phi_DG = v_DG.split()

#Computing reconstruction
v_DG1 = Function(problem.V_DG1)
v_DG1.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
u_DG1,phi_DG1 = v_DG1.split()
strains = problem.strains_2d(u_DG1, phi_DG1)
sig,mu = problem.stresses_2d(strains)
deff,kappa = strains

#ref
U = FunctionSpace(mesh, 'DG', 1)


##plot
#fig = plot(u_DG1[0])
#plt.colorbar(fig)
#plt.show()
#fig = plot(project(u_D[0], U))
#plt.colorbar(fig)
#plt.show()
#fig = plot(u_DG1[1])
#plt.colorbar(fig)
#plt.show()
#fig = plot(project(u_D[1], U))
#plt.colorbar(fig)
#plt.show()
#fig = plot(phi_DG1)
#plt.colorbar(fig)
#plt.show()
#fig = plot(project(phi_D, U))
#plt.colorbar(fig)
#plt.show()

##test defs and stresses
#fig = plot(mu[1])
#plt.colorbar(fig)
#plt.show()
#sys.exit()

#plot errors
#fig = plot(u_DG1[0] - project(u_D[0], U))
#plt.colorbar(fig)
#plt.show()
#fig = plot(u_DG1[1] - project(u_D[1], U))
#plt.colorbar(fig)
#plt.show()
#fig = plot(phi_DG1 - project(phi_D, U))
#plt.colorbar(fig)
#plt.show()
x = SpatialCoordinate(mesh)
#sigg = problem.G*(2*(1-nu)/(1-2*nu)*x[1] + 2*nu/(1-2*nu)*x[0])
sigg = problem.G*((1+a)*x[0] + (1-a)*x[1])
fig = plot(sig[0,1] - project(sigg, U))
plt.colorbar(fig)
plt.show()
sys.exit()

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


