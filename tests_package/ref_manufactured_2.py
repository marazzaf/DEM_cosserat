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

    #print(type(d))
    D = G * as_matrix(d)
    return D

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
    
#mesh
L = 0.5
nb_elt = 80
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
U,S = V.split()

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

def boundary(x, on_boundary):
    return on_boundary

ds = Measure('ds')(subdomain_data=boundary_parts)

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)

D = D_Matrix(G, nu, l, N)

#BC
A = 0.5 #What value to put?
B = 1 #Same question
u_D = Expression(('A*pow(x[0]*x[0]+x[1]*x[1],2)','A*pow(x[0]*x[0]+x[1]*x[1],2)'), A=A, degree=4)
phi_D = Expression('B*(x[0]*x[0]+x[1]*x[1])', B=B, degree=2)
t = Expression(('-G*(8*A*x[0]*(a*x[0]+b*x[1]) + 4*A*(x[0]*x[0]+x[1]*x[1])*(a+c) + 2*x[1]*((4*A*x[0]-B)*d + (4*A*x[1]+B)*c))','-G*(8*A*x[1]*(a*x[1]+b*x[0]) + 4*A*(x[0]*x[0]+x[1]*x[1])*(a+c) + 2*x[0]*((4*A*x[0]-B)*c + (4*A*x[1]+B)*d))'), G=G, A=A, B=B, a=a, b=b, c=c, d=d, degree = 2)
#t = Constant((0,0))
c = Expression('G*((x[0]*x[0]+x[1]*x[1])*(4*A*(x[1]-x[0])+2*B)*(d-c) - 16*l*l*B)', G=G, A=A, B=B, c=c, d=d, l=l, degree = 3)
#c = Constant(0)

#initial lhs and rhs
lhs = inner(strain(v, eta), D*strain(u, psi))*dx
L = inner(t, v)*dx + inner(c, eta)*dx

#BC
bc_u = DirichletBC(U, u_D, boundary)
bc_phi = DirichletBC(S, phi_D, boundary)
bc = [bc_u, bc_phi]

U_h = Function(V)
solve(lhs == L, U_h, bc)
u_h, psi_h = U_h.split()

#Reference solutions
W = VectorFunctionSpace(mesh, 'CG', 2)
u = interpolate(u_D, W)
W = FunctionSpace(mesh, 'CG', 1)
phi = interpolate(phi_D, W)

#solutions and ref
img = plot(u_h[0])
plt.colorbar(img)
plt.show()
fig = plot(u[0])
plt.colorbar(fig)
plt.show()
img = plot(u_h[1])
plt.colorbar(img)
plt.show()
fig = plot(u[1])
plt.colorbar(fig)
plt.show()
img = plot(psi_h)
plt.colorbar(img)
plt.show()
fig = plot(phi)
plt.colorbar(fig)
plt.show()
#sys.exit()

##errors
#img = plot(u_h[0]-u[0])
#plt.colorbar(img)
#plt.show()
#img = plot(u_h[1]-u[1])
#plt.colorbar(img)
#plt.show()
#img = plot(psi_h-phi)
#plt.colorbar(img)
#plt.show()
#sys.exit()

#write convergence test to see if okay...
err_grad = np.sqrt(errornorm(u_h, u, 'H10')**2 + errornorm(psi_h, phi, 'H10')**2)
err_L2 = np.sqrt(errornorm(u_h, u, 'L2')**2 + errornorm(psi_h, phi, 'L2')**2)
print(V.dofmap().global_dimension())
print(err_grad)
print(err_L2)
