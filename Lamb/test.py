#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from petsc4py import PETSc
from ufl import sign
    
# Mesh
Lx,Ly = 4e3,2e3
nb_elt = 5 #50 computation #5 #debug
mesh = RectangleMesh(Point(-Lx/2,0),Point(Lx/2,Ly),int(Lx/Ly)*nb_elt,nb_elt,"crossed")

# Parameters
nu = 0.25 # Poisson's ratio
E = 1.88e10 #Young Modulus
rho = 2200 #volumic mass
G = E/(1+nu) #Shear modulus
Gc = 0
a = Gc/G
l = float(0.5*mesh.hmax()/np.sqrt(2)) # intrinsic length scale

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

# Sub domain for clamp at left end
def top(x, on_boundary):
    return near(x[1], Ly) and on_boundary

top_boundary = AutoSubDomain(top)
top_boundary.mark(boundary_subdomains, 1)

ds = ds(subdomain_data=boundary_parts)

#volume load
x0 = 0
y0 = Ly - 20
sigma = 14.5
r_squared = Expression('pow(x[0]-x0,2) + pow(x[1]-y0,2)', x0=x0, y0=y0, degree=2)
#psi = 1e10 * (1 - 0.5 * r_squared/sigma**2) * exp(-0.5*r_squared/sigma**2) / (np.pi*sigma**4)
#psi = Expression('pow(x[0]-x0,2)+pow(x[1]-y0,2) < 1 ? (1 - t*t/sigma/sigma) * exp(-0.5*t*t/sigma/sigma) : 0', x0=x0, y0=y0, sigma=sigma, t=0, degree = 1)
t = 0.5
psi = (1 - t*t/sigma/sigma) * exp(-0.5*t*t/sigma/sigma)
x = SpatialCoordinate(mesh)
X = (x[1]-y0) / (x[0]-x0)
cos_theta = 1. / sqrt(1 + X**2.) * sign(x[0]-x0)
sin_theta = abs(X) / sqrt(1 + X**2.) * sign(x[1]-y0)
load = psi * as_vector((cos_theta, sin_theta, 0)) 
Rhs = problem.assemble_volume_load(load)

##test load
#U = FunctionSpace(mesh ,'CG', 1)
#img = plot(project(t[1], U))
#plt.colorbar(img)
#plt.show()
#sys.exit()

#compliance tensor
problem.micropolar_constants(E, nu, l/2, a)

# Variational problem
elas = problem.elastic_bilinear_form()
K = elas

#Penalty matrix
K += inner_penalty(problem)

#Nitsche penalty bilinear form
bc = [[0, Constant(0), 0], [1, Constant(0), 0], [2, Constant(0), 0]] #Homogeneous Dirichlet on all boundary apart from top surface
K += lhs_bnd_penalty(problem, boundary_parts, bc)

#Newmark-beta parameters
gamma = 0.5
beta = 0.25

# Time-stepping
T = 1 #second
Nsteps = 50
#dt_ = T/Nsteps
#time = np.linspace(0, T, Nsteps+1)
#test
dt_ = 1e-10
time = np.linspace(0, 50*dt_, Nsteps+1)

# Current (unknown) displacement
u = Function(problem.V_DG, name='disp')
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(problem.V_DG)
v_old = Function(problem.V_DG, name='vel')
a_old = Function(problem.V_DG)
#Necessary
v_DG,psi_DG = TestFunctions(problem.V_DG)

#test
truc = Expression(('x[0] < 0 ? 1 : 0', '0', '0'), x0=x0, Ly=Ly, degree = 1)
v_old = interpolate(truc, problem.V_DG)

#Mass matrix
I = 2/5*l*l
M = mass_matrix(problem, rho, I)
#K += M/beta/dt_**2
K = PETScMatrix(K)
#corresponding rhs
def L(uu, vv, aa):
    aux = (uu+dt_*vv)/beta/dt_**2 + (1-2*beta)/2/beta*aa
    disp = as_vector((aux[0],aux[1]))
    rot = aux[2]
    return rho*inner(disp, v_DG)*dx + rho*I*inner(rot,psi_DG)*dx

#outputs
folder = 'test'
file = File(folder+"/output.pvd")

#test Verlet
M = mass_matrix_vec(problem, rho, I)
for (i, dt) in enumerate(np.diff(time)):
    t = time[i+1]
    print("Time: ", t)

    u.vector()[:] += v_old.vector() * dt_
    a_old.vector()[:] = K*u.vector()
    v_old.vector().set_local(v_old.vector().get_local() - dt_*a_old.vector().get_local() / M.get_local())

    #img = plot(sqrt(u[1]*u[1]+u[0]*u[0]))
    img = plot(sqrt(v_old[1]*v_old[1]+v_old[0]*v_old[0]))
    plt.colorbar(img)
    plt.show()
    #sys.exit()

    #output
    file.write(u, t)
    file.write(v_old, t)

sys.exit()

#implicit integration
for (i, dt) in enumerate(np.diff(time)):
    t = time[i+1]
    print("Time: ", t)
    #psi = (1 - t*t/sigma/sigma) * np.exp(-0.5*t*t/sigma/sigma) / r_squared
    psi = 1e-3
    load = psi * as_vector((cos_theta, sin_theta, 0)) # * psi

    # Solve for new displacement
    Rhs = problem.assemble_volume_load(load)
    res = as_backend_type(assemble(L(u_old, v_old, a_old))) + PETScVector(Rhs)
    res = as_backend_type(assemble(L(u_old, v_old, a_old)))
    solve(K, u.vector(), res, 'mumps')
    
    # Update old fields with new quantities
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()
    a_old.vector()[:] = (u_vec-u0_vec-dt_*v0_vec)/beta/dt_**2 - (1-2*beta)/2/beta*a0_vec
    v_old.vector()[:] = v0_vec + dt_*((1-gamma)*a0_vec + gamma*a_old.vector())
    u_old.vector()[:] = u.vector() #Useful?

    #img = plot(sqrt(u[1]*u[1]+u[0]*u[0]))
    img = plot(sqrt(v_old[1]*v_old[1]+v_old[0]*v_old[0]))
    plt.colorbar(img)
    plt.show()
    #sys.exit()

    #output
    file.write(u, t)
    file.write(v_old, t)


