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
nu = 0.3 # Poisson's ratio
l = 0.2 # intrinsic length scale
N = 0.8 # coupling parameter
G = 100

#Parameters for D_Matrix
a = 2*(1-nu)/(1-2*nu)
b = 2*nu/(1-2*nu)
c = 1/(1-N*N)
d = (1-2*N*N)/(1-N*N)

def D_Matrix(G, nu, l, N):
    return G * as_matrix([[a,0,0,b], [0,c,d,0], [0,d,c,0], [b,0,0,a]])

def strain(v, eta):
    gamma = as_vector([v[0].dx(0), v[1].dx(0) - eta, v[0].dx(1) + eta, v[1].dx(1)])
    kappa = grad(eta)
    return gamma, kappa

def stresses(D,strains):
    gamma,kappa = strains
    sigma = dot(D, gamma)
    mu = 4*G*l*l * kappa
    return sigma, mu
    
# Mesh
L = 0.5
nb_elt = 5
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")

#Creating the DEM problem
problem = DEMProblem(mesh, 2*G, 2*G) #sure about second penalty term? #2*G*l

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

#compliance tensor
D = D_Matrix(G, nu, l, N)

# Variational problem
A = elastic_bilinear_form(problem, D, strain, stresses)

#Penalty matrix
A += inner_penalty_bis(problem)

#rhs
t = Constant((-(a+c),-(a+c),0))
rhs = assemble_volume_load(t, problem)

##Nitsche penalty bilinear form
#A += lhs_nitsche_penalty(problem, bc)
u,phi = TrialFunctions(problem.V_DG1)
v,psi = TestFunctions(problem.V_DG1)
vol = CellVolume(problem.mesh)
hF = FacetArea(problem.mesh)
n = FacetNormal(problem.mesh)
h = vol / hF
strains = strain(u,phi)
stress,couple_stress = stresses(D,strains)
stress = as_tensor(((stress[0],stress[1]), (stress[2],stress[3])))
#Bilinear
bilinear = problem.penalty_u/h * inner(u,v) * ds + problem.penalty_phi/h * inner(phi,psi) * ds + inner(dot(couple_stress,n), psi)*ds + inner(dot(stress,n), v) * ds
Mat = assemble(bilinear)
row,col,val = as_backend_type(Mat).mat().getValuesCSR()
Mat = csr_matrix((val, col, row))
#print(Mat.nonzero())
#sys.exit()
A += problem.DEM_to_DG1.T * Mat * problem.DEM_to_DG1

#Solving linear problem
v = spsolve(A,rhs)
v_h = Function(problem.V_DG1)
v_h.vector().set_local(problem.DEM_to_DG1 * v)
u_h, phi_h = v_h.split()

#assert abs(np.linalg.norm(u_h(0,L))) < abs(np.linalg.norm(u_h(0,0))) / 10
#assert abs(np.linalg.norm(phi_h(0,L))) < abs(np.linalg.norm(phi_h(0,0))) / 100
print(u_h(0,L),phi_h(0,L))
print(u_h(0,0),phi_h(0,0))


fig = plot(u_h[0])
plt.colorbar(fig)
plt.savefig('u_x_15.pdf')
plt.show()
fig = plot(u_h[1])
plt.colorbar(fig)
plt.savefig('u_y_15.pdf')
plt.show()
sys.exit()
