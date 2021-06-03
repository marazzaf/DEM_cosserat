from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define mesh
Lx,Ly,Lz = 1., 0.1, 0.04
#mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 3, 2, 2) #test
#folder = 'test'
mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 60, 10, 5) #fine
folder = 'ref'

# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return near(x[0], Lx) and on_boundary

# Elastic parameters
E = 1e3
nu = 0.3
l = 0.1 * Lx

#other parameters
G = 0.5*E/(1+nu)
Gc = 0.5*G
lmbda = 2*G*nu / (1-2*nu)
M = G * l*l
L = M
Mc = M

# Mass density
rho = Constant(1.0)
I = Constant(2/5*l*l) #Quelle valeur donner à ca ?

# Rayleigh damping coefficients
eta_m = Constant(0.)
eta_k = Constant(0.)

# Generalized-alpha method parameters
alpha_m = Constant(0) #Constant(0.2)
alpha_f = Constant(0) #Constant(0.4)
gamma   = Constant(0.5)
beta    = Constant(0.25)

# Time-stepping parameters
T = 1 #4
dt = 1e-2 #1e-5
Nsteps = int(T / dt) + 1

p0 = 1.
cutoff_Tc = T/5
# Define the loading as an expression depending on t
p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

# Function Space
U = VectorElement("CG", mesh.ufl_cell(), 2) # displacement space
S = VectorElement("CG", mesh.ufl_cell(), 1) # micro rotation space
V = FunctionSpace(mesh, MixedElement(U,S)) # dim 6
print('nb dofs FEM: %i' % V.dofmap().global_dimension())
U, S = V.split()
U_1, U_2, U_3 = U.split()
S_1, S_2, S_3 = S.split()

# Test and trial functions
du = TrialFunction(V)
u_ = TestFunction(V)
# Current (unknown) displacement
u = Function(V)
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Set up boundary condition at left end
zero = Constant((0, 0, 0, 0, 0, 0))
bc = DirichletBC(V, zero, left)

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

# Mass form
def m(w, w_):
    u = as_vector((w[0],w[1],w[2]))
    phi = as_vector((w[3],w[4],w[5]))
    u_ = as_vector((w_[0],w_[1],w_[2]))
    phi_ = as_vector((w_[3],w_[4],w_[5]))
    return rho*inner(u, u_)*dx + rho*I*inner(phi,phi_)*dx 

# Elastic stiffness form
def k(w, w_):
    du = as_vector((w[0],w[1],w[2]))
    dphi = as_vector((w[3],w[4],w[5]))
    u_ = as_vector((w_[0],w_[1],w_[2]))
    phi_ = as_vector((w_[3],w_[4],w_[5]))
    epsilon_u = strain(du, dphi)
    epsilon_v = strain(u_, phi_)
    chi_u = torsion(dphi)
    chi_v = torsion(phi_)

    sigma_u = stress(epsilon_u)
    sigma_v = stress(epsilon_v)
    m_u = couple(chi_u)
    m_v = couple(chi_v)

    return inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx

# Work of external forces
def Wext(u_):
    u_aux = as_vector((u_[0],u_[1],u_[2]))
    return dot(u_aux, p)*dss(3)

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
res = m(a_new, u_) + k(du, u_) - Wext(u_)
#res = m(avg(a_old, a_new, alpha_m), u_) + k(avg(u_old, du, alpha_f), u_) - Wext(u_)
a_form = lhs(res)
L_form = rhs(res)

# Define solver for reusing factorization
K, res = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "mumps")
solver.parameters["symmetric"] = True

# Time-stepping
time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
energies = np.zeros((Nsteps+1, 4))
E_damp = 0
E_ext = 0
xdmf_file = XDMFFile(folder+"/flexion_fine.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False
file = open(folder+'/energies.txt', 'w')
file_disp = open(folder+'/disp.txt', 'w')

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
    print("Time: ", t)

    # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
    p.t = t-float(alpha_f*dt)

    # Solve for new displacement
    res = assemble(L_form)
    bc.apply(res)
    solver.solve(K, u.vector(), res)


    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)

    # Save solution to XDMF format
    if i % 100 == 0:
        xdmf_file.write(u, t)

    p.t = t
    # Record tip displacement and compute energies
    u_tip[i+1] = u(1., 0.05, 0.)[1]
    E_elas = assemble(0.5*k(u_old, u_old))
    E_kin = assemble(0.5*m(v_old, v_old))
    E_ext += assemble(Wext(u-u_old))
    u_old.vector()[:] = u.vector()
    
    E_tot = E_elas+E_kin
    energies[i+1, :] = np.array([E_elas, E_kin, E_tot, E_ext])
    file.write('%.2e %.2e %.2e %.2e %.2e\n' % (t, E_elas, E_kin, E_tot, E_ext))
    file_disp.write('%.2e %.2e\n' % (t, u_tip[i+1]))

file.close()
file_disp.close()

## Plot tip displacement evolution
#plt.figure()
#plt.plot(time, u_tip)
#plt.xlabel("Time")
#plt.ylabel("Tip displacement")
#plt.ylim(-0.5, 0.5)
#plt.show()
#
## Plot energies evolution
#plt.figure()
#plt.plot(time, energies)
#plt.legend(("elastic", "kinetic", "total", "exterior"))
#plt.xlabel("Time")
#plt.ylabel("Energies")
#plt.ylim(0, 0.0011)
#plt.show()
