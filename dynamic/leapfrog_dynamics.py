from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define mesh
Lx,Ly,Lz = 1., 0.1, 0.04
mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 3, 2, 2) #test
computation = 'test'
#mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 60, 10, 5) #fine
#computation = 'FEM'

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

# Time-stepping parameters
T       = 1 #4
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
# Current displacement and velocity
u = Function(V, name='disp')
v = Function(V, name='vel')
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)

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

# Assembling matrices
one = interpolate(Constant((1, 1, 1, 1, 1, 1)), V)
mass = assemble(m(du,u_)) * one.vector()
rigidity = assemble(k(du,u_))

#updating fields
def update_fields(disp, vel, vel_old, b):
    u_old.vector()[:] = u.vector()
    u.vector()[:] = disp.vector() + dt * vel.vector()#[:]
    F = b - rigidity*u.vector()#[:]
    v_old.vector()[:] = v.vector()#[:]
    v.vector()[:] = v_old.vector()[:] + F/mass*dt
    return

#time-step
dt = 5e-7 #similar to DEM
Nsteps = int(T/dt) + 1

# Time-stepping
time = np.linspace(0, T, Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
energies = np.zeros((Nsteps+1, 4))
E_damp = 0
E_ext = 0
xdmf_file = XDMFFile(computation+"/flexion_fine.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False
file = open(computation+'/energies_%s.txt' % computation, 'w')
file_disp = open(computation+'/disp_%s.txt' % computation, 'w')

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
    if i % int(1e4) == 0:
        print("Time: ", t)


    # Forces are evaluated at t_{n+1}
    p.t = t

    #Recompute rhs
    res = assemble(Wext(u_)).get_local()
    #Applying bc
    bc.apply(u.vector())
    bc.apply(v.vector())


    # Update old fields with new quantities
    update_fields(u, v, v_old, res)

    # Save solution to XDMF format
    if i % int(1e4) == 0:
        xdmf_file.write(u, t)
        xdmf_file.write(v, t)

        # Record tip displacement and compute energies
        u_tip[i+1] = u(1., 0.05, 0.)[1]
        E_elas = assemble(0.5*k(u,u))
        v_mid = 0.5*(v+v_old)
        E_kin = assemble(0.5*m(v_mid, v_mid))
        E_ext += assemble(Wext(u-u_old))
        E_tot = E_elas+E_kin
        energies[i+1, :] = np.array([E_elas, E_kin, E_tot, E_ext])
        file.write('%.5e %.5e %.5e %.5e %.5e\n' % (t, E_elas, E_kin, E_tot, E_ext))
        file_disp.write('%.5e %.5e\n' % (t, u_tip[i+1]))

file.close()
file_disp.close()
