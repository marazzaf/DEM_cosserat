#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import sys

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

#penalty terms
penalty_u = 2*G
penalty_phi = 2*G*l*l

# Convergence
h = [15, 30, 50, 70, 90] # mesh density

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
    
mesh = Mesh()
with XDMFFile("hole_plate.xdmf") as infile:
    infile.read(mesh)
hm = mesh.hmax()

U = VectorElement("DP", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("DP", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
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
            return on_boundary and abs(x[1] - plate) < tol

t = Constant((0.0, T))
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

bot_boundary = BotBoundary()
bot_boundary.mark(boundary_parts, 1)
left_boundary = LeftBoundary()
left_boundary.mark(boundary_parts, 2)
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 3)

ds = Measure('ds')(subdomain_data=boundary_parts)

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)
gamma,kappa = strain_bis(u,psi)
D,D_aux = D_Matrix(G, nu, l, N)
sigma,mu = stress((gamma,kappa),D_aux)

hF = CellDiameter(mesh)
n = FacetNormal(mesh)
    
elastic = inner(strain(v, eta), D*strain(u, psi))*dx
inner_pen = penalty_u/hF('+') * inner(jump(u),jump(v)) * dS + penalty_phi/hF('+') * inner(jump(psi),jump(eta)) * dS
inner_consistency = inner(dot(avg(sigma),n('+')), jump(v))*dS + inner(dot(avg(mu),n('+')), jump(eta))*dS + inner(jump(mu,n), avg(eta))*dS + inner(jump(sigma,n), avg(v))*dS
bnd_consistency = inner(dot(sigma,n)[1], v[1])*ds(1) + inner(dot(sigma,n)[0], v[0])*ds(2) + inner(dot(mu,n), eta) * (ds(2) + ds(1))
bnd_pen = penalty_u/hF * u[1] * v[1] * ds(1) + penalty_u/hF * u[0] * v[0] * ds(2) + penalty_phi/hF * inner(psi,eta) * (ds(2) + ds(1))
a = elastic + inner_pen + bnd_pen# + inner_consistency + bnd_consistency
 
L = inner(t, v)*ds(3) 

U_h = Function(V)
problem = LinearVariationalProblem(a, L, U_h) #, bc)
solver = LinearVariationalSolver(problem)
solver.solve()
u_h, psi_h = U_h.split()

#plot(mesh)
#plt.show()
#sys.exit()
img = plot(u_h[0])
plt.colorbar(img)
plt.savefig('DG_u_x_15.pdf')
plt.show()
img = plot(u_h[1])
plt.colorbar(img)
plt.savefig('DG_u_y_15.pdf')
plt.show()
img = plot(psi_h)
plt.colorbar(img)
plt.savefig('DG_phi_15.pdf')
plt.show()
sys.exit()

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

#plt.plot(elements_size, errors, "-*", linewidth=2)
#plt.xlabel("elements size")
#plt.ylabel("error")
#plt.show()
