#coding: utf-8

# Plot the error in SCF depending on the elements size
from dolfin import *
import matplotlib.pyplot as plt
#from solver_3D import computation
# Parameters
R = 10.0 # radius
cube = 100.0 # dim

T = 1.0 # traction force

nu = 0.499 # Poisson's ratio
mu = 1000.0 # shear modulus G
lmbda = ( 2.0 * mu * nu ) / (1.0-2.0*nu) # 1st Lame constant

l = 0.2 # intrinsic length scale
N = 0.93 # coupling parameter

# Analytical solution
def AnalyticalSolution(R, l, nu):
    k = R / l
    eta = 0.2 # ratio of the transverse curvature to the principal curvature
    k_1 = (3.0+eta) / ( 9.0 + 9.0*k + 4.0*k**2 + eta*(3.0 + 3.0*k + k**2) )
    SCF = ( 3.0*(9.0 - 5.0*nu + 6.0*k_1*(1.0-nu)*(1.0+k)) ) / \
          ( 2.0*(7.0 - 5.0*nu + 18.0*k_1*(1.0-nu)*(1.0+k)) )
    return SCF

SCF_a = AnalyticalSolution(R, l, nu)

#Loading mesh
mesh = Mesh()
mesh_num = 3
with XDMFFile("meshes/cube_%i.xdmf" % mesh_num) as infile:
    infile.read(mesh)
hm = mesh.hmax()

# Micropolar elastic constants
alpha = ( mu * N**2 ) / (N**2 - 1.0)
beta = mu * l
gamma = mu * l**2
kappa = gamma

# Strain and torsion
def strain(v, eta):
    strain = nabla_grad(v)
    strain += as_tensor([ [ 0, -eta[2], eta[1] ] , [ eta[2], 0, -eta[0] ] , [ -eta[1], eta[0], 0 ] ] )
    return strain

def torsion(eta):
    return nabla_grad(eta)

# Stress and couple stress
def stress(lmbda, mu, kappa, epsilon):
    stress = as_tensor([ \
                         [lmbda*epsilon[0,0]+(mu+kappa)*epsilon[0,0]+ mu*epsilon[0,0],
                          \
                          (mu+kappa)*epsilon[0,1] + mu*epsilon[1,0], \
                          (mu+kappa)*epsilon[0,2] + mu*epsilon[2,0] ], \
                         [ (mu+kappa)*epsilon[1,0] + mu*epsilon[0,1], \
                           lmbda*epsilon[1,1] + (mu+kappa)*epsilon[1,1] +
                           mu*epsilon[1,1], \
                           (mu+kappa)*epsilon[1,2] + mu*epsilon[2,1] ], \
                         [ (mu+kappa)*epsilon[2,0] + mu*epsilon[0,2], \
                           (mu+kappa)*epsilon[2,1] + mu*epsilon[1,2], \
                           lmbda*epsilon[2,2] + (mu+kappa)*epsilon[2,2] +
                           mu*epsilon[2,2]] ])
    return stress

def couple(alpha, beta, gamma, chi):
    couple = as_tensor([ \
                         [ (alpha + beta + gamma)*chi[0,0], \
                           beta*chi[1,0] + gamma*chi[0,1], \
                           beta*chi[2,0] + gamma*chi[0,2] ], \
                         [ beta*chi[0,1] + gamma*chi[1,0], \
                           (alpha + beta + gamma)*chi[1,1], \
                           beta*chi[2,1] + gamma*chi[1,2] ], \
                         [ beta*chi[0,2] + gamma*chi[2,0], \
                           beta*chi[1,2] + gamma*chi[2,1], \
                           (alpha + beta + gamma)*chi[2,2]] ])
    return couple

# Function Space
U = VectorElement("CG", mesh.ufl_cell(), 2) # displacement space
S = VectorElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S)) # dim 6
print('nb dof FEM: %i' % V.dofmap().global_dimension())
U, S = V.split()
U_1, U_2, U_3 = U.split()
S_1, S_2, S_3 = S.split()

# Boundary conditions
class BotBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1]) < tol
        
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[0]) < tol
            
class FrontBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[2]) < tol
        
class TopBoundary(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-6
        return on_boundary and abs(x[1] - cube) < tol
        
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
        
bot_boundary = BotBoundary()
left_boundary = LeftBoundary()
front_boundary = FrontBoundary()
top_boundary = TopBoundary()
top_boundary.mark(boundary_parts, 1)

ds = Measure('ds')(subdomain_data=boundary_parts)

u_0 = Constant(0.0)

left_U_1 = DirichletBC(U_1, u_0, left_boundary)
left_S_2 = DirichletBC(S_2, u_0, left_boundary)
left_S_3 = DirichletBC(S_3, u_0, left_boundary)

bot_U_2 = DirichletBC(U_2, u_0, bot_boundary)
bot_S_1 = DirichletBC(S_1, u_0, bot_boundary)
bot_S_3 = DirichletBC(S_3, u_0, bot_boundary)

front_U_3 = DirichletBC(U_3, u_0, front_boundary)
front_S_1 = DirichletBC(S_1, u_0, front_boundary)
front_S_2 = DirichletBC(S_2, u_0, front_boundary)

bcs = [left_U_1, left_S_2, left_S_3, bot_U_2, bot_S_1, \
           bot_S_3, front_U_3, front_S_1, front_S_2]

# Variational problem
u, phi = TrialFunctions(V)
v, eta = TestFunctions(V)
epsilon_u = strain(u, phi)
epsilon_v = strain(v, eta)
chi_u = torsion(phi)
chi_v = torsion(eta)

sigma_u = stress(lmbda, mu, kappa, epsilon_u)
sigma_v = stress(lmbda, mu, kappa, epsilon_v)
m_u = couple(alpha, beta, gamma, chi_u)
m_v = couple(alpha, beta, gamma, chi_v)

t = Constant((0.0, T, 0.0))
a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx
L = inner(t, v)*ds(1)

U_h = Function(V)
problem = LinearVariationalProblem(a, L, U_h, bcs)
solver = LinearVariationalSolver(problem)
#solver.parameters['linear_solver'] = 'bicgstab'
#solver.parameters['preconditioner'] = 'petsc_amg'
solver.solve()
u_h, phi_h = U_h.split()

file = File('locking_%i_.pvd' % mesh_num)
file << u_h
file << phi_h
        
epsilon_u_h = strain(u_h, phi_h)
sigma_u_h = stress(lmbda, mu, kappa, epsilon_u_h)
U = FunctionSpace(mesh, 'CG', 1)
sigma_yy = project(sigma_u_h[1,1], U)
file << sigma_yy
SCF = sigma_yy(R, 0, 0)

e = abs(SCF - SCF_a) / SCF_a
print('Ref: %.5e' % SCF_a)
print('Computed: %.5e' % SCF)
print('Error: %.2f' % (100*e))

##Calculer plutôt moyenne de la valeur sur le bord du trou?
#boundary_lines = MeshFunction("size_t", mesh, 1)
#dl = Measure('ds')(subdomain_data=boundary_lines)
#class HoleBoundary(SubDomain):
#    def inside(self, x, on_boundary):
#        tol = 1
#        return on_boundary and x[0]*x[0]+x[2]*x[2] < R*R #abs(x[0]*x[0]+x[2]*x[2]-R*R) < tol
#hole_boundary = HoleBoundary()
#hole_boundary.mark(boundary_lines, 5)
#
###h = MaxCellEdgeLength(problem.mesh)
##h = FacetArea(problem.mesh)
##test = assemble(h * dl(5))
#U = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
#test_func = TestFunction(U)
#length = assemble(test_func[0] * dl(5)).get_local().sum()
#print(length)
#values = assemble(sigma_yy * test_func[0] * dl(5)).get_local().sum()
#print(values/length)
