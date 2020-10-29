#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import sys

# Parameters
d = 2 #2d problem
nu = 0.3 # Poisson's ratio
G = 100 # shear modulus

l = 0.2 # intrinsic length scale
N = 0.8 # coupling parameter
c = l/N

# Convergence
h = [15, 30, 50, 70, 90] # mesh density

elements_size = []
errors = []
SCF_0 = []

# Matrix
def D_Matrix(G, nu, l, N):
    d = np.array([ \
        [(2.0*(1.0 - nu))/(1.0 - 2.0*nu), (2.0*nu)/(1.0 - 2.0 * nu), 0.0,0.0,0.0,0.0], \
        [(2.0*nu)/(1.0 - 2.0*nu), (2.0*(1.0 - nu)) / (1.0 - 2.0*nu), 0.0,0.0,0.0,0.0], \
        [0.0,0.0, 1.0/(1.0 - N**2), (1.0 - 2.0*N**2)/(1.0 - N**2), 0.0,0.0], \
        [0.0,0.0, (1.0 - 2.0*N**2)/(1.0 - N**2), 1.0/(1.0 - N**2), 0.0,0.0], \
        [0.0,0.0,0.0,0.0, 4.0*l**2, 0.0], \
        [0.0,0.0,0.0,0.0,0.0, 4.0*l**2] ])
    
    d *= G

    #print(type(d))
    D = as_matrix(d) #Constant(d)
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

def strain_bis(v, omega):
    gamma = grad(v) + as_tensor(([0.,-omega],[omega,0.]))
    kappa = grad(omega)
    return gamma, kappa

def stress(Tuple):
    gamma,kappa = Tuple
    sigma = K * tr(gamma) * Indentity(d) + 2*G * (sym(gamma) - tr(gamma) * Indentity(d) / 3) + 2*Gc * skew(gamma)
    mu = L * tr(kappa) * Identity(d) + 2*M * (sym(kappa) - tr(kappa) * Identity(d) / 3) + 2*Mc * skew(curvature)
    return sigma,mu
    

#Mesh
L = 0.5
nb_elt = 50
mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S))
U,S = V.split()
U_1, U_2 = U.sub(0), U.sub(1)

#Dirichlet BC
u_0 = Constant((0,0,0))
boundary = Boundary()
boundary.mark(boundary_parts, 1)
bc = DirichletBC(V, u_0, boundary)

#Neumann BC
a = 2*(1-nu)/(1-2*nu)
b = 2*nu/(1-2*nu)
#t = G*Constant((-(a+b),-(a+b))) # load
t = Constant((-1,-1)) # load

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)

D = D_Matrix(G, nu, l, N)
    
A = inner(strain(v, eta), D*strain(u, psi))*dx
L = inner(t, v)*dx

U_h = Function(V)
problem = LinearVariationalProblem(A, L, U_h, bc)
solver = LinearVariationalSolver(problem)
solver.solve()
u_h, psi_h = U_h.split()

#Saving results
#results = File('ref_test_pen.pvd')
#u_h.rename('disp', 'disp')
#results << u_h
#psi_h.rename('rotation', 'rotation')
#results << psi_h
res_mesh = HDF5File(MPI.comm_world, 'ref_mesh_test_pen.hdf5', 'w')
res_mesh.write(mesh , "Mesh")
del res_mesh
res = HDF5File(MPI.comm_world, 'ref_test_pen.hdf5', 'w')
res.write(u_h, 'disp')
res.write(psi_h, 'rotation')
del res

##plot(mesh)
##plt.show()
##sys.exit()
#img = plot(u_h[0])
#plt.colorbar(img)
#plt.savefig('ref_u_x_15.pdf')
#plt.show()
#img = plot(u_h[1])
#plt.colorbar(img)
#plt.savefig('ref_u_y_15.pdf')
#plt.show()
#img = plot(psi_h)
#plt.colorbar(img)
#plt.savefig('ref_phi_15.pdf')
#plt.show()
#sys.exit()