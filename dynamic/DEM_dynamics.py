from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define mesh
Lx,Ly,Lz = 1., 0.1, 0.04
mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 3, 2, 2) #test
#mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 60, 10, 5) #fine

# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return near(x[0], Lx) and on_boundary

def left(x, on_boundary):
    return near(x[0], 0) and on_boundary

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

# Newmark-beta parameters for Crank-Nicholson
gamma   = Constant(0.5)
beta    = Constant(0.25)

# Time-stepping parameters
T       = 4.0
Nsteps  = 50
dt = Constant(T/Nsteps)

p0 = 1.
cutoff_Tc = T/5
# Define the loading as an expression depending on t
p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

#Computing coefficients for Cosserat material
problem.micropolar_constants_3d(E, nu, l, Gc, M, Mc)

# Current (unknown) displacement
u = Function(problem.V_DG)
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(problem.V_DG)
u_old_CR = Function(problem.V_CR)
v_old = Function(problem.V_DG)
a_old = Function(problem.V_DG)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)
left_boundary = AutoSubDomain(left)
left_boundary.mark(boundary_subdomains, 1)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Set up boundary condition at left end
u_0 = Constant(0.0)
bc_1 = [0, u_0, 1]
bc_2 = [1, u_0, 1]
bc_3 = [2, u_0, 1]
bc_4 = [3, u_0, 1]
bc_5 = [4, u_0, 1]
bc_6 = [5, u_0, 1]
bcs = [bc_1, bc_2, bc_3, bc_4, bc_5, bc_6]

# Mass form
#mass = mass_matrix(problem, rho, I)
def m(w, w_):
    u = as_vector((w[0],w[1],w[2]))
    phi = as_vector((w[3],w[4],w[5]))
    u_ = as_vector((w_[0],w_[1],w_[2]))
    phi_ = as_vector((w_[3],w_[4],w_[5]))
    return rho*inner(u, u_)*dx + rho*I*inner(phi,phi_)*dx 

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
    
    return (inner(epsilon_v, sigma_u)  + inner(chi_v, m_u))*dx

# Work of external forces
def Wext(u_):
    u_aux = as_vector((u_[0],u_[1],u_[2]))
    return dot(u_aux, p)*dss(3)

## Work of external forces
#Wext = assemble_boundary_load(problem, 3, boundary_subdomains, p)

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old):
    dt_ = float(dt)
    beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old):
    dt_ = float(dt)
    gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    u_old.vector()[:] = u.vector()

# Residual
du_DG = TrialFunction(problem.V_DG)
u_DG = TestFunction(problem.V_DG)
du_CR = TrialFunction(problem.V_CR)
u_CR = TestFunction(problem.V_CR)
a_new = update_a(du_DG, u_old, v_old, a_old)
v_new = update_v(a_new, u_old, v_old, a_old)

#mass
res_mass = m(a_new, u_DG)
a_form_m = lhs(res_mass)
L_form_m = rhs(res_mass)
K_m, res_m = assemble_system(a_form_m, L_form_m)
K_m = as_backend_type(K_m).mat()

#rigidity
res_rigidity = k(du_CR, u_CR) - Wext(u_CR)
a_form_r = lhs(res_rigidity)
L_form_r = rhs(res_rigidity)
K_r, res_r = assemble_system(a_form_r, L_form_r)
K_r = as_backend_type(K_r).mat()

#Nitsche penalty
K_p = lhs_bnd_penalty(problem, boundary_subdomains, bcs)

#define lhs and rhs
K = K_r * problem.DEM_to_CR
K = PETScMatrix(problem.DEM_to_CR.transpose(PETSc.Mat()) * K + K_m + K_p) #ajouter la matrice de pénalisation

# Time-stepping
time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
energies = np.zeros((Nsteps+1, 4))
E_ext = 0
xdmf_file = XDMFFile("ref/flexion.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

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
    p.t = t

    # Solve for new displacement
    res_r = PETScVector(problem.DEM_to_CR.transpose(PETSc.Mat()) * as_backend_type(assemble(L_form_r)).vec())
    res_m = assemble(L_form_m)
    res = res_m + res_r
    solve(K, u.vector(), res, 'mumps')


    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)

    # Save solution to XDMF format
    xdmf_file.write(u, t)

    # Record tip displacement and compute energies
    u_tip[i+1] = u(1., 0.05, 0.)[1]
    E_elas = assemble(0.5*k(u_old, u_old))
    E_kin = assemble(0.5*m(v_old, v_old))
    E_ext += assemble(Wext(u-u_old))
    u_old.vector()[:] = u.vector()
    E_tot = E_elas+E_kin
    energies[i+1, :] = np.array([E_elas, E_kin, E_tot, E_ext])

# Plot tip displacement evolution
plt.figure()
plt.plot(time, u_tip)
plt.xlabel("Time")
plt.ylabel("Tip displacement")
plt.ylim(-0.5, 0.5)
plt.show()

# Plot energies evolution
plt.figure()
plt.plot(time, energies)
plt.legend(("elastic", "kinetic", "total", "exterior"))
plt.xlabel("Time")
plt.ylabel("Energies")
plt.ylim(0, 0.0011)
plt.show()
