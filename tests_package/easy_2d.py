#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *

# Parameters
d = 2 #2d problem
R = 10.0 # radius
plate = 100.0 # plate dimension
nu = 0.3 # Poisson's ratio
G = 1000.0 # shear modulus

l = 0.2 # intrinsic length scale
N = 0.8 # coupling parameter
T = 1.0 # load
c = l/N

#for other version
K = 16.67e3
G = 10e3
Gc = 5e3
L = 10 #pas de valeur
R = 0.01
h3 = 2/5
M = G*R*R/h3
Mc = M

# Convergence
h = 15

elements_size = []
errors = []
SCF_0 = []

# Analytical solution
def AnalyticalSolution(nu, l, c, R):
    
    # Modified Bessel function of the second kind of real order v :
    from scipy.special import kv
    F = 8.0*(1.0-nu)*((l**2)/(c**2)) * \
        1.0 / (( 4.0 + ((R**2)/(c**2)) + \
        ((2.0*R)/c) * kv(0,R/c)/kv(1,R/c) ))
    
    SCF = (3.0 + F) / (1.0 + F) # stress concentration factor

    return SCF

SCF = AnalyticalSolution(nu, l, c, R)

## Matrix
#def D_Matrix(G, nu, l, N):
#    d = G * np.array([ \
#        [(2.0*(1.0 - nu))/(1.0 - 2.0*nu), (2.0*nu)/(1.0 - 2.0 * nu), 0.0,0.0,0.0,0.0], \
#        [(2.0*nu)/(1.0 - 2.0*nu), (2.0*(1.0 - nu)) / (1.0 - 2.0*nu), 0.0,0.0,0.0,0.0], \
#        [0.0,0.0, 1.0/(1.0 - N**2), (1.0 - 2.0*N**2)/(1.0 - N**2), 0.0,0.0], \
#        [0.0,0.0, (1.0 - 2.0*N**2)/(1.0 - N**2), 1.0/(1.0 - N**2), 0.0,0.0], \
#        [0.0,0.0,0.0,0.0, 4.0*l**2, 0.0], \
#        [0.0,0.0,0.0,0.0,0.0, 4.0*l**2] ])
#    
#
#    D = as_matrix(d)
#    return D

def D_Matrix(G, nu, l, N):
    a = 2*(1-nu)/(1-2*nu)
    b = 2*nu/(1-2*nu)
    c = 1/(1-N*N)
    d = (1-2*N*N)/(1-N*N)
    
    D = as_matrix([[a,0,0,b], [b,0,0,a], [0,c,d,0], [0,d,c,0]])
    return G * D

## Strain
#def strain(v, eta):
#    strain = as_vector([ \
#                         v[0].dx(0),
#                         v[1].dx(1),
#                         v[1].dx(0) - eta,
#                         v[0].dx(1) + eta,
#                         eta.dx(0),
#                         eta.dx(1)])
#
#    return strain

def strain_bis(v, omega):
    gamma = grad(v) + as_tensor(([0.,-omega],[omega,0.]))
    kappa = grad(omega)
    return gamma, kappa

def stess(D,strains):
    gamma,kappa = strains
    sigma = dot(D, gamma)
    mu = 4*G*l*l * kappa
    return sigma, mu

#def stress(Tuple):
#    gamma,kappa = Tuple
#    sigma = K * tr(gamma) * Indentity(d) + 2*G * (sym(gamma) - tr(gamma) * Indentity(d) / 3) + 2*Gc * skew(gamma)
#    mu = L * tr(kappa) * Identity(d) + 2*M * (sym(kappa) - tr(kappa) * Identity(d) / 3) + 2*Mc * skew(curvature)
#    return sigma,mu
    
    
# Mesh
geometry = Rectangle(Point(0,0),Point(plate, plate)) - \
           Circle(Point(0,0), R, h)
mesh = generate_mesh(geometry, h)
hm = mesh.hmax()

#U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
#S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
#V = FunctionSpace(mesh, MixedElement(U,S))
#U,S = V.split()
#U_1, U_2 = U.sub(0), U.sub(1)

#Creating the DEM problem
problem = DEMProblem(mesh, d)

#Creating the graph
G = connectivity_graph(problem)

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
        return on_boundary and abs(x[1] - plate) < tol

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
bot_U_2 = DirichletBC(U_2, u_0, bot_boundary)
left_S = DirichletBC(S, u_0, left_boundary)
bot_S = DirichletBC(S, u_0, bot_boundary)

bc = [left_U_1, bot_U_2, left_S, bot_S]

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)

D = D_Matrix(G, nu, l, N)

#test forme varia... Fonctionne!
truc = strain_bis(v,eta)
ttruc = strain_bis(u, psi)
ttruc = stress(D,ttruc)
a = (inner(truc[0],ttruc[0]) + inner(truc[1],ttruc[1])) * dx

#rhs
L = inner(t, v)*ds(1)

U_h = Function(V)
problem = LinearVariationalProblem(a, L, U_h, bc)
solver = LinearVariationalSolver(problem)
solver.solve()
u_h, psi_h = U_h.split()

# Stress
epsilon = strain(u_h, psi_h)
sigma = D*epsilon
sigma_yy = project(sigma[1])
#Other version
#epsilon = strain_bis(u_h, psi_h)
#sigma = stress(epsilon)[0]
#sigma_yy = project(sigma[1])

error = abs((sigma_yy(10.0, 1e-6) - SCF) / SCF)

elements_size.append(hm)
SCF_0.append(sigma_yy(10.0, 1e-6))
errors.append(error)
        
print("Analytical SCF: %.5e" % SCF)
print(elements_size)
print(errors)
print(SCF_0)


file = File("sigma.pvd")
file << sigma_yy

plt.plot(elements_size, errors, "-*", linewidth=2)
plt.xlabel("elements size")
plt.ylabel("error")
plt.show()
