#coding: utf-8

# Computation of the solution in Cosserat elasticity
from dolfin import *

Lz = 6e-2
Lx,Ly = Lz/10,Lz/10
nb_elt = 5 #to start
mesh = BoxMesh(Point(-Lx/2,-Ly/2,0), Point(Lx/2,Ly/2,Lz), nb_elt, nb_elt, 10*nb_elt)

# Micropolar elastic constants
E = 3e9
nu = 0.3
G = 0.5*E/(1+nu)
lmbda = 2*G*nu / (1-2*nu)
Gc = 0.84e9
l = 1e-2
M = 2*G * l*l
L = 2/3*M
Mc = M

#Loading parameter
T = 1e9

# Strain and torsion
def strain(v, eta):
    strain = nabla_grad(v)
    strain += as_tensor([ [ 0, -eta[2], eta[1] ] , [ eta[2], 0, -eta[0] ] , [ -eta[1], eta[0], 0 ] ] )
    return strain

def torsion(eta):
    return nabla_grad(eta)

# Stress and couple stress
def stress(e):
    return lmbda * tr(e) * Identity(3) + 2*G * sym(e) + 2*Gc * skew(e)

def couple(kappa):
    return L * tr(kappa) * Identity(3) + 2*M * sym(kappa) + 2*Mc * skew(kappa)

# Function Space
U = VectorElement("CG", mesh.ufl_cell(), 2) # displacement space
S = VectorElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S)) # dim 6
print('nb dofs FEM: %i' % V.dofmap().global_dimension())
U, S = V.split()
U_1, U_2, U_3 = U.split()
S_1, S_2, S_3 = S.split()

# Boundary conditions
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[2]) < tol

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[2] - Lz) < tol

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

left_boundary = LeftBoundary()
right_boundary = RightBoundary()
right_boundary.mark(boundary_parts, 1)

ds = Measure('ds')(subdomain_data=boundary_parts)

u_0 = Constant(0.0)

left_U_1 = DirichletBC(U_1, u_0, left_boundary)
left_S_2 = DirichletBC(S_2, u_0, left_boundary)
left_S_3 = DirichletBC(S_3, u_0, left_boundary)

left_U_2 = DirichletBC(U_2, u_0, left_boundary)
left_S_1 = DirichletBC(S_1, u_0, left_boundary)
left_S_3 = DirichletBC(S_3, u_0, left_boundary)

left_U_3 = DirichletBC(U_3, u_0, left_boundary)
left_S_1 = DirichletBC(S_1, u_0, left_boundary)
left_S_2 = DirichletBC(S_2, u_0, left_boundary)

bcs = [left_U_1, left_S_2, left_S_3, left_U_2, left_S_1, \
       left_S_3, left_U_3, left_S_1, left_S_2]

# Variational problem
u, phi = TrialFunctions(V)
v, eta = TestFunctions(V)
epsilon_u = strain(u, phi)
epsilon_v = strain(v, eta)
chi_u = torsion(phi)
chi_v = torsion(eta)

sigma_u = stress(epsilon_u)
sigma_v = stress(epsilon_v)
m_u = couple(chi_u)
m_v = couple(chi_v)

t = Constant((0.0, T, 0.0))
a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx
L = inner(t, eta)*ds(1) #change that for a flexion. Put a momentum.

U_h = Function(V)
problem = LinearVariationalProblem(a, L, U_h, bcs)
solver = LinearVariationalSolver(problem)
solver.parameters['linear_solver'] = 'mumps'
solver.solve()
u_h, phi_h = U_h.split()

#output ref
file = File('FEM/computation_%i_.pvd' % nb_elt)
file << u_h
file << phi_h

##output stress
#epsilon_u_h = strain(u_h, phi_h)
#sigma_u = stress(epsilon_u_h)
#U = FunctionSpace(mesh, 'CG', 1)
#sigma_yy = project(sigma_u[1,1], U)
#file << sigma_yy
