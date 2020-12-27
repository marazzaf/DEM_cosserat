from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from slepc4py import SLEPc

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define mesh
Lx,Ly,Lz = 1., 0.1, 0.04
mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 3, 2, 2) #test
#mesh = BoxMesh(Point(0., 0., 0.), Point(Lx, Ly, Lz), 60, 10, 5) #fine

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

# Time-stepping parameters
T       = 4.
p0 = 1.
cutoff_Tc = T/5
# Define the loading as an expression depending on t
p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

#Creating the DEM problem
cte = 10
problem = DEMProblem(mesh, cte)

#Computing coefficients for Cosserat material
problem.micropolar_constants(E, nu, l, Gc, M)

# Current (unknown) displacement
u = Function(problem.V_DG) #disp,rot
v = Function(problem.V_DG) #vel,rot vel
# Fields from previous time step (velocity)
v_old = Function(problem.V_DG)

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
mass,min_mass = mass_matrix(problem, rho, I)

#Rigidity matrix
K = problem.elastic_bilinear_form()
#Nitsche penalty bilinear form
K += lhs_bnd_penalty(problem, boundary_subdomains, bcs)
#Penalty matrix
K += inner_penalty(problem)

# Work of external forces
Wext = assemble_boundary_load(problem, 3, boundary_subdomains, p)

#converting matrix
A = K.tocsr()
petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices,A.data))
K = PETScMatrix(petsc_mat)

#Computing time-step
E = SLEPc.EPS(); E.create() #creating Eigenvalue solver
E.setOperators(petsc_mat)
E.setProblemType(SLEPc.EPS.ProblemType.HEP)
E.setFromOptions()
E.solve() #solving
assert E.getConverged() #otherwise did not converge
vr, wr = petsc_mat.getVecs()
vi, wi = petsc_mat.getVecs()
dt = 2 * np.sqrt(min_mass / E.getEigenpair(0, vr, vi).real)
Nsteps = int(T/dt) + 1

sys.exit()

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
    p.t = t-float(alpha_f*dt)

    # Solve for new displacement
    res = assemble(L_form)
    bc.apply(res)
    solve(K, v_DG.vector(), res, 'mumps')


    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)

    # Save solution to XDMF format
    xdmf_file.write(u, t)

    p.t = t
    # Record tip displacement and compute energies
    u_tip[i+1] = u(1., 0.05, 0.)[1]
    E_elas = assemble(0.5*k(u_old, u_old))
    E_kin = assemble(0.5*m(v_old, v_old))
    E_ext += assemble(Wext(u-u_old))
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
