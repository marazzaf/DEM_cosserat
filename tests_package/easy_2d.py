#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve

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

def D_Matrix(G, nu, l, N):
    a = 2*(1-nu)/(1-2*nu)
    b = 2*nu/(1-2*nu)
    c = 1/(1-N*N)
    d = (1-2*N*N)/(1-N*N)
    
    D = as_matrix([[a,0,0,b], [b,0,0,a], [0,c,d,0], [0,d,c,0]])
    return G * D

def strain(v, eta):
    gamma = as_vector([v[0].dx(0), v[1].dx(0) - eta, v[0].dx(1) + eta, v[1].dx(1)])
    kappa = grad(eta)
    return gamma, kappa

def stress(D,strains):
    gamma,kappa = strains
    sigma = dot(D, gamma)
    mu = 4*G*l*l * kappa
    return sigma, mu
    
# Mesh
mesh = Mesh()
with XDMFFile("hole_plate.xdmf") as infile:
    infile.read(mesh)

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l) #sure about second penalty term?

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
bot_boundary.mark(boundary_parts, 2)
left_boundary = LeftBoundary()
left_boundary.mark(boundary_parts, 3)
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

#ds = Measure('ds')(subdomain_data=boundary_parts)

u_0 = Constant(0.0)
bc_1 = [[0], [u_0], 3]
bc_2 = [[1], [u_0], 2]
bc_3 = [[2], [u_0], 3]
bc_4 = [[2], [u_0], 2]
bc_bis = [bc_1, bc_2, bc_3, bc_4]

#compliance tensor
D = D_Matrix(G, nu, l, N)

# Variational problem
A = elastic_bilinear_form(problem, D, strain, stress)

#rhs
rhs = assemble_boundary_load(problem, 1, boundary_parts, t)

#Imposing weakly the BC!
rhs += rhs_nitsche_penalty(problem, bc_bis, D, strain, stress)

#Nitsche penalty bilinear form
A += lhs_nitsche_penalty(problem, bc_bis)

#Penalty matrix
#A += inner_penalty(problem)
A += inner_penalty_bis(problem)

#Solving linear problem
v = spsolve(A,rhs)
v_h = Function(problem.V_DG)
v_h.vector().set_local(v)
u_h, psi_h = v_h.split()

fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_15.pdf')
plt.show()
fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_15.pdf')
plt.show()
fig = plot(psi_h)
plt.colorbar(fig)
plt.savefig('phi_15.pdf')
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

plt.plot(elements_size, errors, "-*", linewidth=2)
plt.xlabel("elements size")
plt.ylabel("error")
plt.show()
