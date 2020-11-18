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
from scipy.sparse.linalg import spsolve,cg

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
with XDMFFile("hole_plate_very_fine.xdmf") as infile: #fine
#with XDMFFile("hole_plate_fine.xdmf") as infile: #fine
#with XDMFFile("hole_plate.xdmf") as infile:
    infile.read(mesh)

#Creating the DEM problem
problem = DEMProblem(mesh, 4*G, 4*G*l*l) #sure about second penalty term?
print('nb dof DEM: %i' % problem.nb_dof_DEM)

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
b += rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
A += lhs_bnd_penalty(problem, boundary_parts, bc)

#Penalty matrix
A += inner_penalty_light(problem)
#A += inner_consistency(problem)

#Solving linear problem
v = spsolve(A,b)
#v,info = cg(A,b)
#assert info == 0
v_DG = Function(problem.V_DG)
v_DG.vector().set_local(v)
u_DG, psi_DG = v_DG.split()
v_DG1 = Function(problem.V_DG1)
v_DG1.vector().set_local(problem.DEM_to_DG1 * v)
u_DG1, psi_DG1 = v_DG1.split()

fig = plot(u_DG[0])
plt.colorbar(fig)
#plt.savefig('u_x_15.pdf')
plt.show()
fig = plot(u_DG[1])
plt.colorbar(fig)
#plt.savefig('u_y_15.pdf')
plt.show()
fig = plot(psi_DG)
plt.colorbar(fig)
#plt.savefig('phi_15.pdf')
plt.show()

# Stress
strains = problem.strains(u_DG1, psi_DG1)
sigma,mu = problem.stresses(strains)
U = FunctionSpace(mesh, 'DG', 0)
sigma_yy = project(sigma[1], U)


error = abs((sigma_yy(10.0, 1e-6) - SCF) / SCF)
        
print("Analytical SCF: %.5e" % SCF)
print('Computed SCF: %.5e' % sigma_yy(10.0, 1e-6))
print(error)


file = File("sigma_very_fine.pvd")
file << sigma_yy
file << u_DG
file << u_DG1
