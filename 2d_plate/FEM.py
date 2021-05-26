#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import matplotlib.pyplot as plt
import sys

T = 1
G = 1e3
nu = 0.3
r = 0.864e-3
l = r / 1
a = 1/3

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
    mu = 4*G*l*l * kappa
    return sigma, mu

mesh = Mesh()
with XDMFFile("mesh/hole_plate_4.xdmf") as infile:
    infile.read(mesh)
    
#Functionnal spaces
U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
print('nb dof CG: %i' % V.dofmap().global_dimension())
U,S = V.split()
U_1, U_2 = U.sub(0), U.sub(1)

# Boundary conditions
class BotBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1]) < tol

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[0]) < tol

class TopBoundary(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1] - 16.2e-3) < tol

t = Constant((0.0, T))
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

bot_boundary = BotBoundary()
left_boundary = LeftBoundary()
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

ds = Measure('ds')(subdomain_data=boundary_parts)

u_0 = Constant(0.0)
left_U_1 = DirichletBC(U.sub(0), u_0, left_boundary)
bot_U_2 = DirichletBC(U.sub(1), u_0, bot_boundary)
left_S = DirichletBC(S, u_0, left_boundary)
bot_S = DirichletBC(S, u_0, bot_boundary)

bc = [left_U_1, bot_U_2, left_S, bot_S]

# Variational problem
u, phi = TrialFunctions(V)
v, psi = TestFunctions(V)
e,kappa = strain(u,phi)
sigma,mu = stress(e,kappa)
e,kappa = strain(v,psi)

A = inner(sigma, e)*dx + inner(mu, kappa)*dx
L = inner(t, v)*ds(1)

#Solving problem
U_h = Function(V)
solve(A == L, U_h, bc)
u_h, psi_h = U_h.split()

#Computing max stress
e,kappa = strain(u_h,psi_h)
sigma,mu = stress(e,kappa)
W = FunctionSpace(mesh, 'DG', 0)
sig = project(sigma[1,1], W)
print(max(sig.vector().get_local()))

##Plot
#img = plot(sig)
#plt.colorbar(img)
#plt.show()


