#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
    
# Mesh
h = 1e-3
nb_elt = 10
mesh = RectangleMesh(Point(-h/2,0),Point(h/2,h),nb_elt,5*nb_elt,"crossed")

# Parameters
nu = 0 #0.3 # Poisson's ratio #Correct?
l = 1e-4 # intrinsic length scale
a = 2 #2 #or 1???
G = 10e9 #Shear modulus
Gc = 20e9
M = G*l*l
E =  2*G*(1+nu) #Young Modulus

#Creating the DEM problem
def strain(v,psi):
    e = grad(v) + as_tensor(((0, 1), (-1, 0))) * psi
    kappa = grad(psi)
    return e,kappa

def stress(e, kappa):
    eps = as_vector((e[0,0], e[1,1], e[0,1], e[1,0]))
    aux_1 = 2*(1-nu)/(1-2*nu)
    aux_2 = 2*nu/(1-2*nu)
    Mat = G * as_tensor(((aux_1,aux_2,0,0), (aux_2, aux_1,0,0), (0,0,1+a,1-a), (0,0,1-a,1+a)))
    sig = dot(Mat, eps)
    sigma = as_tensor(((sig[0], sig[2]), (sig[3], sig[1])))
    mu = M * kappa
    return sigma, mu

#boundary
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

#boundaries
def top(x, on_boundary):
    return near(x[1], h) and on_boundary

def down(x, on_boundary):
    return near(x[1], 0) and on_boundary

# Sub domain for rotation at right end
def left_right(x, on_boundary):
    return (near(x[0], -h/2) or near(x[0], h/2)) and on_boundary

top_boundary = AutoSubDomain(top)
top_boundary.mark(boundary_parts, 1)
left_right_boundary = AutoSubDomain(left_right)
left_right_boundary.mark(boundary_parts, 2)
down_boundary = AutoSubDomain(down)
down_boundary.mark(boundary_parts, 3)
ds = ds(subdomain_data=boundary_parts)

#ref solution
delta = 2*np.sqrt(G*Gc/((G+Gc)*M))
mat = np.array([[np.exp(delta*h)-np.exp(-delta*h), -0.5/G], [-2*Gc/(G+Gc)/delta*(np.exp(delta*h)-np.exp(-delta*h)), h/G]])
vec = np.array([-0.1,0.01*h])
res = np.linalg.solve(mat,vec)
K1 = res[0]
K2 = -K1
K3 = 0
tau_c = res[1]
omega21 = Gc/(G+Gc)*(K1+K2) - 0.5*tau_c/G

#Functionnal spaces
U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
print('nb dof CG: %i' % V.dofmap().global_dimension())
U,S = V.split()
U_1, U_2 = U.sub(0), U.sub(1)

#Dirichlet BC
t_U1 = DirichletBC(U_1, Constant(1e-5), top_boundary)
d_U1 = DirichletBC(U_1, Constant(0), down_boundary)
d_U2 = DirichletBC(U_2, Constant(0), down_boundary)
d_S = DirichletBC(S, Constant(omega21), down_boundary)
t_S = DirichletBC(S, Constant(-0.1), top_boundary)
lr_U = DirichletBC(U_2, Constant(0), left_right_boundary)
bcs = [t_U1, d_U1, d_U2, t_S, d_S, lr_U]

# Variational problem
u, phi = TrialFunctions(V)
v, psi = TestFunctions(V)
e,kappa = strain(u,phi)
sigma,mu = stress(e,kappa)
e,kappa = strain(v,psi)
A = inner(sigma, e)*dx + inner(mu, kappa)*dx
t = Constant((0,0))
L = inner(t, v) * dx


#Solving linear problem
sol = Function(V)
print('Solve!')
solve(A == L, sol, bcs=bcs)
u,phi = sol.split()

##plot
#fig = plot(u[0])
#plt.colorbar(fig)
#plt.show()
#fig = plot(phi)
#plt.colorbar(fig)
#plt.show()
#sys.exit()

#computed rotation
U = FunctionSpace(mesh, 'CG', 1)
xx = np.arange(0, h, h/20)
rot = np.zeros_like(xx)
disp = np.zeros_like(xx)
for i,X in enumerate(xx):
    rot[i] = phi(0,X)
    disp[i] = u(0,X)[0]

##plot ref rotation
xxx = np.arange(0, h, 1e-6)
omega = K1*np.exp(delta*xxx) + K2*np.exp(-delta*xxx) - 0.5*tau_c/G
plt.plot(xxx, omega, '-', label='analytical')
plt.plot(xx, rot, '*', label='computed')
plt.xlim((0, h))
plt.ylim((-0.1, 0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.title('Rotation')
plt.ylabel(r'$\varphi(x_2)$')
plt.xlabel(r'$x_2$')
plt.legend(loc='lower left')
plt.savefig('rotation_FEM.pdf')
plt.show()

##plot ref rotation
u_0 = -2*Gc/(G+Gc)*(K1/delta*np.exp(delta*xxx) - K2/delta*np.exp(-delta*xxx)) + tau_c/G*xxx + K3
plt.plot(xxx, u_0, '-', label='analytical')
plt.plot(xx, disp, '*', label='computed')
plt.xlim((0, h))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ylabel(r'$u_1(x_2)$')
plt.title('Vertical displacement')
plt.xlabel(r'$x_2$')
plt.legend(loc='upper left')
plt.savefig('disp_FEM.pdf')
plt.show()
