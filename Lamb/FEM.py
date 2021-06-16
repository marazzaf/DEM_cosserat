from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import mpi4py

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["allow_extrapolation"] = True

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

# Mesh
Lx,Ly = 4e3,2e3
nb_elt = 10 #100 computation #5 #debug
mesh = RectangleMesh(Point(-Lx/2,0),Point(Lx/2,Ly),int(Lx/Ly)*nb_elt,nb_elt,"crossed")
folder = 'FEM_test'

# Parameters
nu = 0.25 # Poisson's ratio
E = 1.88e10 #Young Modulus
rho = 2200 #volumic mass
G = E/(1+nu) #Shear modulus
Gc = 0
a = Gc/G
h = mesh.hmax()
l = float(0.5*h/np.sqrt(2)) # intrinsic length scale
I = Constant(2/5*l*l)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

# Sub domain for clamp at left end
def top(x, on_boundary):
    return near(x[1], Ly) and on_boundary

def down(x, on_boundary):
    return near(x[1], 0) and on_boundary

top_boundary = AutoSubDomain(top)
top_boundary.mark(boundary_parts, 1)
down_boundary = AutoSubDomain(down)
down_boundary.mark(boundary_parts, 2)
ds = ds(subdomain_data=boundary_parts)


# Generalized-alpha method parameters
alpha_m = Constant(0.)
alpha_f = Constant(0.)
gamma   = Constant(0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)

# Time-stepping parameters
T = 0.1 #1
Nsteps = 50
dt = Constant(T/Nsteps)

#Ricker wavelet loading
#position of pulse
x0 = 0
y0 = Ly/2 #Ly - 20
#frequence
sigma = 14.5
domain = Expression('pow(x[0]-x0,2) + pow(x[1]-y0,2) < Lx*Lx/100 ? 1 : 0', h=h, x0=x0, y0=y0, Lx=Lx, degree=2)
psi = Expression('2/sqrt(3*sigma)/pow(pi,0.25)*(1 - t*t/sigma/sigma) * exp(-0.5*t*t/sigma/sigma)', sigma=sigma, t=0, degree = 1)
#load = psi * domain * Constant((0,-1,0))
load = E * Constant((0,-1,0))

# Function Space
U = VectorElement("CG", mesh.ufl_cell(), 2) # displacement space
S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S)) # dim 6
if rank == 0:
    print('nb dofs FEM: %i' % V.dofmap().global_dimension())
U, S = V.split()

# Test and trial functions
du = TrialFunction(V)
u_ = TestFunction(V)
# Current (unknown) displacement
u = Function(V, name='disp')
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V, name='vel')
a_old = Function(V)

# Set up boundary condition at left end
zero = Constant((0, 0, 0))
bc = DirichletBC(V, zero, boundary_parts, 2)

#Elastic terms
def strain(v,psi):
    e = grad(v) + as_tensor(((0, 1), (-1, 0))) * psi
    kappa = grad(psi)
    return e,kappa

def stress(e, kappa):
    eps = as_vector((e[0,0], e[1,1], e[0,1], e[1,0]))
    aux_1 = 2*(1-nu)/(1-2*nu)
    aux_2 = 2*nu/(1-2*nu)
    Mat = G * as_tensor(((aux_1,aux_2,0,0), (aux_2, aux_1,0,0), (0,0,1+a,1-a), (0,0,1-a,1+a)))
    sig = dot(Mat, eps)
    sigma = as_tensor(((sig[0], sig[2]), (sig[3], sig[1])))
    mu = 4*G*l*l * kappa
    return sigma, mu

# Mass form
def m(w, w_):
    du = as_vector((w[0],w[1]))
    dphi = w[2]
    u_ = as_vector((w_[0],w_[1]))
    phi_ = w_[2]
    return rho*inner(du, u_)*dx + rho*I*dphi*phi_*dx 

# Elastic stiffness form
def k(w, w_):
    du = as_vector((w[0],w[1]))
    dphi = w[2]
    u_ = as_vector((w_[0],w_[1]))
    phi_ = w_[2]
    epsilon_u,chi_u = strain(du, dphi)
    epsilon_v,chi_v = strain(u_, phi_)

    sigma_u,m_u = stress(epsilon_u,chi_u)
    sigma_v,m_v = stress(epsilon_v,chi_v)

    return inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx

# Work of external forces
def Wext(u_):
    return dot(u_, load) * dx

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    #u_old.vector()[:] = u.vector()

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
res = m(avg(a_old, a_new, alpha_m), u_) + k(avg(u_old, du, alpha_f), u_) - Wext(u_)
a_form = lhs(res)
L_form = rhs(res)

# Define solver for reusing factorization
K, res = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True

# Time-stepping
time = np.linspace(0, T, Nsteps+1)
E_ext = 0
file = XDMFFile(comm,folder+"/FEM_lamb.xdmf")
#file = File(folder+"/FEM_lamb.pvd")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.parameters["rewrite_function_mesh"] = False
file_en = open(folder+'/energies.txt', 'w', 1)

def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

for (i, dt) in enumerate(np.diff(time)):

    t = time[i+1]
    if rank == 0:
        print("Time: ", t)

    # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
    psi.t = t-float(alpha_f*dt)

    # Solve for new displacement
    res = assemble(L_form)
    bc.apply(res)
    mtf = res.get_local()
    print(mtf[mtf.nonzero()])
    solver.solve(u.vector(), res)

    img = plot(u[1])
    plt.colorbar(img)
    plt.show()
    U = FunctionSpace(mesh, 'CG', 1)
    file.write(local_project(u[1], U), t)


    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)

    # Save solution to XDMF format
    #file.write(u, t)
    #file.write(v_old, t)

    # Record tip displacement and compute energies
    E_elas = assemble(0.5*k(u, u))
    E_kin = assemble(0.5*m(v_old, v_old))
    E_ext += assemble(Wext(u-u_old))
    u_old.vector()[:] = u.vector()
    
    E_tot = E_elas+E_kin
    file_en.write('%.2e %.2e %.2e %.2e %.2e\n' % (t, E_elas, E_kin, E_tot, E_ext))

file_en.close()
