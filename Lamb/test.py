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
Lx,Ly = 4e3,2e3
nb_elt = 10
mesh = RectangleMesh(Point(-Lx/2,0),Point(Lx/2,Ly),Lx/Ly*nb_elt,nb_elt,"crossed")

# Parameters
nu = 0.25 # Poisson's ratio
E =  1.88e10 #Young Modulus
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

#BC
u_D = Expression(('1e-5*x[1]/h','0'), h=h, degree=1)
phi_D = Expression('-0.1*x[1]/h + omega21', h=h, omega21=omega21, degree=1)

#volume load
t = Expression(('', '', ''), degree=5)
Rhs = problem.assemble(t) #continue that!

#compliance tensor
problem.micropolar_constants(E, nu, l/2, a)

# Variational problem
elas = problem.elastic_bilinear_form()
Lhs = elas

#Penalty matrix
inner_pen = inner_penalty(problem) #test
Lhs += inner_pen

#Listing Dirichlet BC
bc = [[0,u_D[0],1], [1, u_D[1],1], [2, phi_D,1], [1, Constant(0), 2]]

#Nitsche penalty rhs
nitsche_and_bnd = rhs_bnd_penalty(problem, boundary_parts, bc) 
Rhs += nitsche_and_bnd

#Nitsche penalty bilinear form
bnd = lhs_bnd_penalty(problem, boundary_parts, bc)
Lhs += bnd

#Solving linear problem
v_DG = Function(problem.V_DG)
print('Solve!')
solve(PETScMatrix(Lhs), v_DG.vector(), PETScVector(Rhs), 'mumps')
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
plt.plot(xxx, omega, '-', label='analytical')
plt.plot(xx, rot, '*', label='computed')
plt.xlim((0, h))
plt.ylim((-0.1, 0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.title('Vertical displacement')
plt.ylabel(r'$u_1$')
plt.xlabel(r'$x_2$')
plt.legend(loc='lower left')
plt.savefig('disp.pdf')
plt.show()

##plot ref rotation
u_0 = -2*Gc/(G+Gc)*(K1/delta*np.exp(delta*xxx) - K2/delta*np.exp(-delta*xxx)) + tau_c/G*xxx + K3
plt.plot(xxx, u_0, '-', label='analytical')
plt.plot(xx, disp, '*', label='computed')
plt.xlim((0, h))
#plt.ylim((0, 4e-6))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ylabel(r'$\varphi$')
plt.title('Rotation')
plt.xlabel(r'$x_2$')
plt.legend(loc='upper left')
plt.savefig('rotation.pdf')
plt.show()
