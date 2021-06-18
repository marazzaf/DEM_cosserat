#coding: utf-8

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from petsc4py import PETSc
from ufl import sign
#from scipy.sparse.linalg import eigsh

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
    
# Mesh
Lx,Ly = 2e3,1e3
nb_elt = 100 #100 computation #5 #debug
mesh = RectangleMesh(Point(-Lx/2,0),Point(Lx/2,Ly),int(Lx/Ly)*nb_elt,nb_elt,"crossed")
folder = 'test' #'big' #'test'

# Parameters
nu = 0.25 # Poisson's ratio
E = 1.88e10 #Young Modulus
rho = 2200 #volumic mass
G = 0.5*E/(1+nu) #Shear modulus
Gc = G
a = Gc/G
h = mesh.hmax()
l = float(0.5*h/np.sqrt(2)) # intrinsic length scale

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

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

#volume load
x0 = 0
y0 = Ly - 100
sigma = 14.5
radius = 50
domain = Expression('sqrt(pow(x[0]-x0,2) + pow(x[1]-y0,2)) < radius ? 1 : 0', radius=radius, x0=x0, y0=y0, degree=2)
psi = Expression('2/sqrt(3*sigma)/pow(pi,0.25)*(1 - t*t/sigma/sigma) * exp(-0.5*t*t/sigma/sigma)', sigma=sigma, t=0, degree = 1)
load = psi * domain * Constant((0,-1,0))
Rhs = problem.assemble_volume_load(load)

#compliance tensor
problem.micropolar_constants(E, nu, l/2, a)

# Variational problem
elas = problem.elastic_bilinear_form()
K = elas

#Penalty matrix
K += inner_penalty(problem)

#Nitsche penalty bilinear form
bc = [[0, Constant(0), 2], [1, Constant(0), 2], [2, Constant(0), 2], [0, Constant(0), 0]] #Homogeneous Dirichlet on bottom boundary surface
K += lhs_bnd_penalty(problem, boundary_parts, bc)

#Newmark-beta parameters
gamma = 0.5
beta = 0.25

# Current (unknown) displacement
u = Function(problem.V_DG, name='disp')
u_DG1 = Function(problem.V_DG1, name='disp DG1')
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(problem.V_DG)
v_old = Function(problem.V_DG, name='vel')
v_old_DG1 = Function(problem.V_DG1, name='vel DG1')
a_old = Function(problem.V_DG)

# Time-stepping implicit
T = 0.5 #second
Nsteps = 10000
dt_ = T/Nsteps
time = np.linspace(0, T, Nsteps+1)

#Mass matrix
I = 2/5*l*l
M = mass_matrix(problem, rho, I)
m1 = 1/beta/dt_/dt_
m2 = 1/beta/dt_
m3 = 0.5/beta - 1
K_ref = K.copy()
K += m1*M

#damping (penalty on velocity)
h = CellDiameter(mesh)
v_DG1,psi_DG1 = TestFunctions(problem.V_DG1)
u_DG1_,phi_DG1_ = TrialFunctions(problem.V_DG1)
res_pv = 4*G * (inner(v_DG1,u_DG1_) + l*l*psi_DG1*phi_DG1_) / h * ds(2)
K_pv = problem.DEM_to_DG1.transpose(PETSc.Mat()) * as_backend_type(assemble(res_pv)).mat() * problem.DEM_to_DG1
c1 = gamma/beta/dt_
c2 = -(1 - gamma/beta)
c3 = -dt_ * (1 - 0.5*gamma/beta)
K += c1*K_pv
K = PETScMatrix(K)

#outputs
file = XDMFFile(folder+"/output.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.parameters["rewrite_function_mesh"] = False
file_en = open(folder+'/energies.txt', 'w', 1)

#implicit integration
E_ext = 0
for (i, dt) in enumerate(np.diff(time)):
    t = time[i+1]
    print("Time: ", t)
    psi.t = t
    load = psi * domain * Constant((0,-1,0))

    # Solve for new displacement
    Rhs = problem.assemble_volume_load(load)
    res = PETScVector(Rhs) + PETScMatrix(M) * (m1*u_old.vector()+m2*v_old.vector()+m3*a_old.vector()) + PETScMatrix(K_pv) * (c1*u_old.vector()+c2*v_old.vector()+c3*a_old.vector())
    solve(K, u.vector(), res, 'mumps')
    
    # Update old fields with new quantities
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()
    a_old.vector()[:] = (u_vec-u0_vec-dt_*v0_vec)/beta/dt_**2 - (1-2*beta)/2/beta*a0_vec
    v_old.vector()[:] = v0_vec + dt_*((1-gamma)*a0_vec + gamma*a_old.vector())

    ##img = plot(sqrt(u[1]*u[1]+u[0]*u[0]))
    #img = plot(sqrt(v_old[1]*v_old[1]+v_old[0]*v_old[0]))
    #plt.colorbar(img)
    #plt.show()
    ##sys.exit()

    #output
    file.write(u, t)
    file.write(v_old, t)

    #Check energy balance
    E_elas = 0.5 * u.vector().inner(PETScMatrix(K_ref)*u.vector())
    E_kin = 0.5 * v_old.vector().inner(PETScMatrix(M)*v_old.vector())
    E_ext += PETScVector(Rhs).inner(u.vector()-u_old.vector())
    u_old.vector()[:] = u.vector()
    E_tot = E_elas+E_kin
    file_en.write('%.2e %.2e %.2e %.2e %.2e\n' % (t, E_elas, E_kin, E_tot, E_ext))


file_en.close()
