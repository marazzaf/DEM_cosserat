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
    

#for hx in h :
    
    ## Mesh
    #geometry = Rectangle(Point(0,0),Point(plate, plate))-Circle(Point(0,0), R, hx)
    #mesh = generate_mesh(geometry, hx)
mesh = Mesh()
#with XDMFFile("hole_plate.xdmf") as infile:
with XDMFFile("hole_plate_fine.xdmf") as infile:
#with XDMFFile("hole_plate_very_fine.xdmf") as infile:
    infile.read(mesh)
hm = mesh.hmax()

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
            return on_boundary and abs(x[1] - plate) < tol

t = Constant((0.0, T))
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

bot_boundary = BotBoundary()
left_boundary = LeftBoundary()
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

ds = Measure('ds')(subdomain_data=boundary_parts) #Measure("ds")

u_0 = Constant(0.0)
left_U_1 = DirichletBC(U.sub(0), u_0, left_boundary)
bot_U_2 = DirichletBC(U.sub(1), u_0, bot_boundary)
left_S = DirichletBC(S, u_0, left_boundary)
bot_S = DirichletBC(S, u_0, bot_boundary)

bc = [left_U_1, bot_U_2, left_S, bot_S]

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)

D = D_Matrix(G, nu, l, N)
    
a = inner(strain(v, eta), D*strain(u, psi))*dx
L = inner(t, v)*ds(1)

U_h = Function(V)
problem = LinearVariationalProblem(a, L, U_h, bc)
solver = LinearVariationalSolver(problem)
solver.solve()
u_h, psi_h = U_h.split()

#plot(mesh)
#plt.show()
#sys.exit()
img = plot(u_h[0])
plt.colorbar(img)
plt.savefig('ref_u_x.pdf')
plt.show()
img = plot(u_h[1])
plt.colorbar(img)
plt.savefig('ref_u_y.pdf')
plt.show()
img = plot(psi_h)
plt.colorbar(img)
plt.savefig('ref_phi.pdf')
plt.show()

# Stress
epsilon = strain(u_h, psi_h)
sigma = D*epsilon
sigma_yy = project(sigma[1])

error = abs((sigma_yy(10.0, 1e-6) - SCF) / SCF)
        
print("Analytical SCF: %.5e" % SCF)
print('Computed SCF: %.5e' % sigma_yy(10.0, 1e-6))
print(error)


file = File("sigma_very_fine.pvd")
file << sigma_yy
file << project(psi_h)

#plt.plot(elements_size, errors, "-*", linewidth=2)
#plt.xlabel("elements size")
#plt.ylabel("error")
#plt.show()
