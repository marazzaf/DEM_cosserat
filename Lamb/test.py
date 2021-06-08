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
nb_elt = 5 #5 #debug
mesh = RectangleMesh(Point(-Lx/2,0),Point(Lx/2,Ly),int(Lx/Ly)*nb_elt,nb_elt,"crossed")

# Parameters
nu = 0.25 # Poisson's ratio
E = 1.88e10 #Young Modulus
rho = 2200 #volumic mass
G = E/(1+nu) #Shear modulus
Gc = 0
a = Gc/G
l = 0.5*mesh.hmax()/np.sqrt(2) # intrinsic length scale

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
ds = ds(subdomain_data=boundary_parts)

#volume load
x0 = 0
y0 = Ly - 20
r_squared = Expression('pow(x[0]-x0,2) + pow(x[1]-y0,2)', x0=x0, y0=y0, degree=2)
sigma = 14.5
t = (1 - 0.5 * r_squared/sigma**2) * exp(-0.5*r_squared/sigma**2) / (np.pi*sigma**4)
x = SpatialCoordinate(mesh)
X = (x[1]-y0) / (x[0]-x0)
cos_theta = 1. / sqrt(1 + X**2.) * sign(x[0]-x0) #check signs
sin_theta = abs(X) / sqrt(1 + X**2.) * sign(x[1]-y0) #check signs
t = as_vector((t*cos_theta, t*sin_theta, 0)) 
Rhs = problem.assemble_volume_load(t) #continue that!

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
K += inner_penalty(problem) #test

#Nitsche penalty bilinear form
K += lhs_bnd_penalty(problem, boundary_parts)

#Mass matrix
I = 2/5*l*l
M = mass_matrix(problem, rho, I)
K += M
#(u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

#Newmark-beta parameters
gamma = Constant(0.5)
beta = Constant(0.25)

# Current (unknown) displacement
u = Function(problem.V_DG)
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(problem.V_DG)
v_old = Function(problem.V_DG)
a_old = Function(problem.V_DG)

# Time-stepping
T = 1 #second
Nsteps  = 50
#dt = Constant(T/Nsteps)
time = np.linspace(0, T, Nsteps+1)

for (i, dt) in enumerate(np.diff(time)):

    t = time[i+1]
    print("Time: ", t)

    # Solve for new displacement
    res_r = PETScVector(problem.DEM_to_CR.transpose(PETSc.Mat()) * as_backend_type(assemble(L_form_r)).vec())
    res_m = assemble(L_form_m)
    res = res_r + res_m + Rhs
    solve(K, u.vector(), res, 'mumps')
    
    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)


