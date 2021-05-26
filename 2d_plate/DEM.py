#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *

# Parameters
nu = 0.3 # Poisson's ratio
G = 1e3 # shear modulus
E = 2*G*(1+nu)
a = 1/3
r = 0.864e-3
l = r / 1 # intrinsic length scale
T = 1.0 # load
    
# Mesh
mesh = Mesh()
with XDMFFile("mesh/hole_plate_5.xdmf") as infile:
    infile.read(mesh)

#Creating the DEM problem
cte = 1
problem = DEMProblem(mesh, cte) #1e3 semble bien
#print('nb dof DEM: %i' % problem.nb_dof_DEM)

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

#For BC
bot_boundary = BotBoundary()
bot_boundary.mark(boundary_parts, 2)
left_boundary = LeftBoundary()
left_boundary.mark(boundary_parts, 3)
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

bc = [[0, Constant(0), 3], [2, Constant(0), 3], [1, Constant(0), 2], [2, Constant(0), 2]]

#For compliance tensor
problem.micropolar_constants(E, nu, l, a)

# Variational problem
A = problem.elastic_bilinear_form()

#rhs
b = assemble_boundary_load(problem, 1, boundary_parts, t)

#Imposing weakly the BC!
#b += rhs_bnd_penalty(problem, boundary_parts, bc)

#Nitsche penalty bilinear form
A += lhs_bnd_penalty(problem, boundary_parts, bc)

#Penalty matrix
A += inner_penalty(problem) #light
#A += inner_consistency(problem)

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(PETScMatrix(A), v_DG.vector(), PETScVector(b), 'mumps')
u_DG,phi_DG = v_DG.split()

#Computing reconstruction
v_DG1 = Function(problem.V_DG1)
v_DG1.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
u_DG1,phi_DG1 = v_DG1.split()

#Computing max stress
strains = problem.strains_2d(u_DG1,phi_DG1)
sigma,mu = problem.stresses_2d(strains)
W = FunctionSpace(mesh, 'DG', 0)
sig = project(sigma[1,1], W)
print(max(sig.vector().get_local()))

#Plot
img = plot(sig)
plt.colorbar(img)
plt.show()

#Reconstruction of facets
U_CR = FunctionSpace(mesh, 'CR', 1)
h = CellDiameter(mesh)
v_CR = TestFunction(U_CR)

truc = inner(avg(v_CR), avg(sig)+cte*G/h('+')*jump(u_DG1[1])) / h('+') * dS + inner(v_CR, sig) / h * ds
sig_avg = as_backend_type(assemble(truc))
print(max(sig_avg.get_local()))
