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
    
# Mesh
h = 1e-3
nb_elt = 10
mesh = RectangleMesh(Point(-h/2,0),Point(h/2,h),nb_elt,5*nb_elt,"crossed")

# Parameters
nu = 0 #0.3 # Poisson's ratio #Correct?
l = 1e-4 # intrinsic length scale
a = 2 #2 #or 1???
G = 10e9 #Shear modulus
Gc = 20e9
M = G*l*l
E =  2*G*(1+nu) #Young Modulus

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

#boundaries
def top_down(x, on_boundary):
    return (near(x[1], 0) or near(x[1], h)) and on_boundary

# Sub domain for rotation at right end
def left_right(x, on_boundary):
    return (near(x[0], -h/2) or near(x[0], h/2)) and on_boundary

top_down_boundary = AutoSubDomain(top_down)
top_down_boundary.mark(boundary_parts, 1)
left_right_boundary = AutoSubDomain(left_right)
left_right_boundary.mark(boundary_parts, 2)
ds = ds(subdomain_data=boundary_parts)

#ref solution
delta = 2*np.sqrt(G*Gc/((G+Gc)*M))
mat = np.array([[np.exp(delta*h)-np.exp(-delta*h), -0.5/G], [-2*Gc/(G+Gc)/delta*(np.exp(delta*h)-np.exp(-delta*h)), h/G]])
vec = np.array([-0.1,0.01*h])
res = np.linalg.solve(mat,vec)
K1 = res[0]
K2 = -K1
K3 = 0
tau_c = res[1]
omega21 = Gc/(G+Gc)*(K1+K2) - 0.5*tau_c/G

#BC
u_D = Expression(('1e-5*x[1]/h','0'), h=h, degree=1)
phi_D = Expression('-0.1*x[1]/h + omega21', h=h, omega21=omega21, degree=1)

#compliance tensor
problem.micropolar_constants(E, nu, l/2, a)

# Variational problem
elas = problem.elastic_bilinear_form()
lhs = elas

#Penalty matrix
inner_pen = inner_penalty(problem) #test
lhs += inner_pen

#Listing Dirichlet BC
bc = [[0,u_D[0],1], [1, u_D[1],1], [2, phi_D,1], [1, Constant(0), 2]]

#Nitsche penalty rhs
nitsche_and_bnd = rhs_bnd_penalty(problem, boundary_parts, bc) 
Rhs = nitsche_and_bnd

#Nitsche penalty bilinear form
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
lhs += bnd

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(PETScMatrix(lhs), v_DG.vector(), PETScVector(Rhs), 'mumps')
u_DG,phi_DG = v_DG.split()

#Computing reconstruction
v_DG1 = Function(problem.V_DG1)
v_DG1.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
u_DG1,phi_DG1 = v_DG1.split()

##plot
#fig = plot(u_DG1[0])
#plt.colorbar(fig)
#plt.show()
#fig = plot(phi_DG1)
#plt.colorbar(fig)
#plt.show()
#sys.exit()

#computed rotation
U = FunctionSpace(mesh, 'CG', 1)
xx = np.arange(0, h, h/20)
rot = np.zeros_like(xx)
disp = np.zeros_like(xx)
for i,X in enumerate(xx):
    rot[i] = phi_DG1(0,X)
    disp[i] = u_DG1(0,X)[0]

##plot ref rotation
xxx = np.arange(0, h, 1e-6)
omega = K1*np.exp(delta*xxx) + K2*np.exp(-delta*xxx) - 0.5*tau_c/G
plt.plot(xxx, omega, '-')
plt.plot(xx, rot, '*')
plt.xlim((0, h))
plt.ylim((-0.1, 0))
plt.show()

##plot ref rotation
u_0 = -2*Gc/(G+Gc)*(K1/delta*np.exp(delta*xxx) - K2/delta*np.exp(-delta*xxx)) + tau_c/G*xxx + K3
plt.plot(xxx, u_0, '-')
plt.plot(xx, disp, '*')
plt.xlim((0, h))
#plt.ylim((0, 4e-6))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.show()
