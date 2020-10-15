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
    
# Mesh
mesh = Mesh()
with XDMFFile("hole_plate.xdmf") as infile:
    infile.read(mesh)

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G*l*l) #sure about second penalty term?

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

bc = [[0, Constant(0), 3], [1, Constant(0), 2], [2, Constant(0), 3], [2, Constant(0), 3]]

#compliance tensor
problem.D = problem.D_Matrix(G, nu, N, l)

# Variational problem
A = problem.elastic_bilinear_form()

#rhs
b = assemble_boundary_load(problem, 1, boundary_parts, t)

#Imposing weakly the BC!
b += rhs_nitsche_penalty(problem, bc)

#Nitsche penalty bilinear form
A += lhs_nitsche_penalty(problem, bc)

#Penalty matrix
A += inner_penalty(problem)

#Solving linear problem
v = spsolve(A,b)
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
