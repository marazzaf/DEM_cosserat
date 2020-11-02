#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
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
with XDMFFile("meshes/hole_plate_3.xdmf") as infile:
    infile.read(mesh)
hm = mesh.hmax()

U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
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
bot_boundary.mark(boundary_parts, 2)
left_boundary = LeftBoundary()
left_boundary.mark(boundary_parts, 3)
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

ds = Measure('ds')(subdomain_data=boundary_parts)

# Variational problem
u, psi = TrialFunctions(V)
v, eta = TestFunctions(V)

D,D_aux = D_Matrix(G, nu, l, N)

a = inner(strain(v, eta), D*strain(u, psi))*dx
L = inner(t, v)*ds(1)

#For Nitsche penalty
#rhs_nitsche = inner(dot(D*strain(v, eta), n), u_D) * ds
n = FacetNormal(mesh)
trial_strain = strain_bis(u, psi)
test_strain = strain_bis(v, eta)
trial_stress,trial_couple_stress = stress(trial_strain, D_aux)
test_stress,test_couple_stress = stress(test_strain, D_aux)
lhs_nitsche = -inner(dot(trial_stress, n)[0], v[0]) * ds(3) - inner(dot(trial_stress, n)[1], v[1]) * ds(2) - inner(dot(trial_couple_stress, n), eta) * (ds(2) + ds(3)) + inner(dot(test_stress, n)[0], u[0]) * ds(3) + inner(dot(test_stress, n)[1], u[1]) * ds(2) + inner(dot(test_couple_stress, n), psi) * (ds(2) + ds(3))
a += lhs_nitsche

#u_D = Constant(1)
#rhs_nitsche = inner(dot(test_stress, n)[0], u_D) * ds(3) + inner(dot(test_stress, n)[1], u_D) * ds(2)# + inner(dot(test_couple_stress, n), eta) * (ds(2) + ds(3))
#L += rhs_nitsche

U_h = Function(V)
#problem = LinearVariationalProblem(a, L, U_h)
#solver = LinearVariationalSolver(problem)
#solver.solve()
solve(a == L, U_h)
u_h, psi_h = U_h.split()

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

## Stress
#epsilon = strain(u_h, psi_h)
#sigma = D*epsilon
#sigma_yy = project(sigma[1])
#sigma_yy.set_allow_extrapolation(True)
#
#error = abs((sigma_yy(10.0, 1e-6) - SCF) / SCF)
#        
##print("Analytical SCF: %.5e" % SCF)
#print(S.dofmap().global_dimension())
#print(error)
#print(SCF)

#file = File("sigma.pvd")
#file << sigma_yy

ref_mesh = Mesh()
with XDMFFile("meshes/hole_plate_4.xdmf") as infile:
    infile.read(ref_mesh)
#rotations
xdmf = XDMFFile(MPI.comm_world, 'ref_rot.xdmf')
V_ref = FunctionSpace(ref_mesh,'CG',1)
ref_rot = Function(V_ref)
xdmf.read_checkpoint(ref_rot, 'rot', 0)
#disp
xdmf = XDMFFile(MPI.comm_world, 'ref_disp.xdmf')
V_ref = VectorFunctionSpace(ref_mesh,'CG',2)
ref_disp = Function(V_ref)
xdmf.read_checkpoint(ref_disp, 'disp', 0)

ref_disp.set_allow_extrapolation(True)
ref_disp = project(ref_disp, VectorFunctionSpace(mesh, 'CG', 2))
ref_rot.set_allow_extrapolation(True)
ref_rot = project(ref_rot, FunctionSpace(mesh, 'CG', 1))
#error_disp = u_h - ref_disp
#img = plot(sqrt(error_disp[0]**2 + error_disp[1]**2))
#plt.colorbar(img)
#plt.show()

#write convergence test to see if okay...
err_grad = np.sqrt(errornorm(u_h, ref_disp, 'H10')**2 + errornorm(psi_h, ref_rot, 'H10')**2)
err_L2 = np.sqrt(errornorm(u_h, ref_disp, 'L2')**2 + errornorm(psi_h, ref_rot, 'L2')**2)
print(V.dofmap().global_dimension())
print(err_grad)
print(err_L2)
