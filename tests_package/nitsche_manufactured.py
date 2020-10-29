#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

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

# Matrix
def D_Matrix(G, nu, l, N):
    d = np.array([ \
        [(2.0*(1.0 - nu))/(1.0 - 2.0*nu), (2.0*nu)/(1.0 - 2.0 * nu), 0.0,0.0,0.0,0.0], \
        [(2.0*nu)/(1.0 - 2.0*nu), (2.0*(1.0 - nu)) / (1.0 - 2.0*nu), 0.0,0.0,0.0,0.0], \
        [0.0,0.0, 1.0/(1.0 - N**2), (1.0 - 2.0*N**2)/(1.0 - N**2), 0.0,0.0], \
        [0.0,0.0, (1.0 - 2.0*N**2)/(1.0 - N**2), 1.0/(1.0 - N**2), 0.0,0.0], \
        [0.0,0.0,0.0,0.0, 4.0*l**2, 0.0], \
        [0.0,0.0,0.0,0.0,0.0, 4.0*l**2] ])
    
    d_aux = np.array([ \
        [(2*(1 - nu))/(1 - 2*nu), (2*nu)/(1 - 2*nu), 0.,0.], \
        [(2*nu)/(1 - 2*nu), (2*(1 - nu)) / (1 - 2*nu), 0.,0.], \
        [0.,0., 1/(1 - N**2), (1 - 2*N**2)/(1 - N**2)], \
        [0.,0., (1 - 2*N**2)/(1 - N**2), 1/(1 - N**2)] ])

    #print(type(d))
    D = G * as_matrix(d)
    D_aux = G * as_tensor(d_aux)
    return D,D_aux

# Strain
def strain(v, eta):
    strain = as_vector([ \
                         v[0].dx(0),
                         v[1].dx(1),
                         v[1].dx(0) - eta,
                         v[0].dx(1) + eta,
                         eta.dx(0),
                         eta.dx(1)])

    return strain

def strain_bis(v, eta):
    gamma = as_vector([v[0].dx(0), v[1].dx(1), v[1].dx(0) - eta, v[0].dx(1) + eta])
    kappa = grad(eta)
    return gamma, kappa

def stress(Tuple, D):
    gamma,kappa = Tuple
    sigma = dot(D, gamma)
    sigma = as_tensor( ([sigma[0],sigma[3]],[sigma[2],sigma[1]]) )
    mu = 4*G*l*l * kappa
    return sigma,mu
    
#mesh
L = 0.5
nb_elt = 25
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
U,S = V.split()
U_1, U_2 = U.sub(0), U.sub(1)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

ds = Measure('ds')(subdomain_data=boundary_parts)

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)

D,D_aux = D_Matrix(G, nu, l, N)

#BC
A = 0.5 #What value to put?
B = 1 #Same question
u_D = Expression(('A*(x[0]*x[0]+x[1]*x[1])','A*(x[0]*x[0]+x[1]*x[1])'), A=A, degree=2)
phi_D = Expression('B*(x[0]-x[1])', B=B, degree=1)
t = Expression(('-(A*2*(a+d)+B*(c-d))','-(A*2*(a+d)+B*(c-d))'), A=A, B=B, a=a, b=b, c=c, d=d, degree = 1)
c = Expression('2*(x[0]-x[1])*(d-c)*(B-A)', A=A, B=B, c=c, d=d, degree = 1)

#initial lhs and rhs
lhs = inner(strain(v, eta), D*strain(u, psi))*dx
L = inner(t, v)*dx + inner(c, eta)*dx

#For Nitsche penalty
n = FacetNormal(mesh)
trial_strain = strain_bis(u, psi)
test_strain = strain_bis(v, eta)
trial_stress,trial_couple_stress = stress(trial_strain, D_aux)
test_stress,test_couple_stress = stress(test_strain, D_aux)
lhs_nitsche = -inner(dot(trial_stress, n), v) * ds - inner(dot(trial_couple_stress, n), eta) * ds + inner(dot(test_stress, n), u) * ds + inner(dot(test_couple_stress, n), psi) * ds
lhs += lhs_nitsche

#rhs Nitsche penalty
rhs_nitsche = inner(dot(test_stress, n), u_D) * ds + inner(dot(test_couple_stress, n), phi_D) * ds
L += rhs_nitsche

U_h = Function(V)
problem = LinearVariationalProblem(lhs, L, U_h)
solver = LinearVariationalSolver(problem)
solver.solve()
u_h, psi_h = U_h.split()

#Reference solutions
U = VectorFunctionSpace(mesh, 'CG', 2)
u = interpolate(u_D, U)
U = FunctionSpace(mesh, 'CG', 1)
phi = interpolate(phi_D, U)

#img = plot(u_h[0]-u[0])
#plt.colorbar(img)
#plt.show()
img = plot(u_h[0])
plt.colorbar(img)
plt.show()
fig = plot(u[0])
plt.colorbar(fig)
plt.show()

#img = plot(u_h[1]-u[1])
#plt.colorbar(img)
#plt.show()
img = plot(u_h[1])
plt.colorbar(img)
plt.show()
fig = plot(u[1])
plt.colorbar(fig)
plt.show()

#img = plot(psi_h-phi)
#plt.colorbar(img)
#plt.show()
img = plot(psi_h)
plt.colorbar(img)
plt.show()
fig = plot(phi)
plt.colorbar(fig)
plt.show()
