#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
from scipy.sparse.linalg import spsolve
import pytest #for unit tests

# Parameters
nu = 0.3 # Poisson's ratio
l = 0.2 # intrinsic length scale
N = 0.8 # coupling parameter
G = 100

#Parameters for D_Matrix
a = 2*(1-nu)/(1-2*nu)
b = 2*nu/(1-2*nu)
c = 1/(1-N*N)
d = (1-2*N*N)/(1-N*N)
    
# Mesh
L = 0.5
nb_elt = 10
#mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
@pytest.mark.parametrize("mesh", [RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")]) #, BoxMesh(Point(-L, -L, -L), Point(L, L, L), nb_elt, nb_elt, nb_elt)])

def test_pen_bnd(mesh):

    #Creating the DEM problem
    problem = DEMProblem(mesh, 2*G, 2*G*l*l)
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

    #compliance tensor
    problem.D = problem.D_Matrix(G, nu, N, l)

    # Variational problem
    A = problem.elastic_bilinear_form()

    #Penalty matrix
    A += inner_penalty_light(problem)

    #rhs
    t = Constant((-(a+b),-(a+b),0))
    rhs = problem.assemble_volume_load(t)

    #Nitsche penalty bilinear form. Homogeneous Dirichlet in this case.
    #A += lhs_nitsche_penalty(problem, strain, stresses)
    bc = [[0, Constant(0)], [1, Constant(0)], [2, Constant(0)]]
    #bc = [[0, Constant(1)], [1, Constant(1)], [2, Constant(1)]]
    A += lhs_bnd_penalty(problem, boundary_parts, bc)
    rhs += rhs_bnd_penalty(problem, boundary_parts, bc)
    #A += lhs_bnd_penalty(problem)

    #Solving linear problem
    v = spsolve(A,rhs)
    v_h = Function(problem.V_DG1)
    v_h.vector().set_local(problem.DEM_to_DG1 * v)
    u_h, phi_h = v_h.split()

    F = FacetArea(problem.mesh)
    W = FunctionSpace(problem.mesh, 'CR', 1)
    w = TestFunction(W)
    aire = sum(assemble(w * ds).get_local())
    assert round(assemble(u_h[0] / aire * ds), 2) == 0
    assert round(assemble(u_h[1] / aire * ds), 2) == 0
    assert round(assemble(phi_h / aire * ds), 2) == 0

##Loading ref
#ref_mesh = HDF5File(MPI.comm_world, 'ref_mesh_test_pen.hdf5', 'r')
#mesh_ref = Mesh()
#ref_mesh.read(mesh_ref, "Mesh", False)
#print('mesh ok')
#U_ref = VectorFunctionSpace(mesh_ref, 'DG', 1)
#u_ref = Function(U_ref)
#ref = HDF5File(MPI.comm_world, 'ref_test_pen.hdf5', 'r')
#ref.read(u_ref, 'disp')
#print('disp ok')
#PHI_ref = FunctionSpace(mesh_ref, 'DG', 1)
#psi_ref = Function(PHI_ref)
#ref.read(psi_ref, 'rotation')


#assert abs(np.linalg.norm(u_h(0,L))) < abs(np.linalg.norm(u_h(0,0))) / 10
#assert abs(np.linalg.norm(phi_h(0,L))) < abs(np.linalg.norm(phi_h(0,0))) / 100
#print(u_h(0,L),phi_h(0,L))
#print(u_h(0,0),phi_h(0,0))

#gamma,kappa = strain(u_h,phi_h)
#U = FunctionSpace(problem.mesh, 'DG', 0)
#fig = plot(local_project(gamma[3], U))
#plt.colorbar(fig)
#plt.savefig('stress_1_1.pdf')
#plt.show()
#fig = plot(local_project(kappa[1],U))
#plt.colorbar(fig)
#plt.savefig('kappa_1.pdf')
#plt.show()
#sys.exit()

#fig = plot(u_h[0])
#plt.colorbar(fig)
##plt.savefig('u_x_25.pdf')
#plt.show()
#fig = plot(u_h[1])
#plt.colorbar(fig)
##plt.savefig('u_y_25.pdf')
#plt.show()
#fig = plot(phi_h)
#plt.colorbar(fig)
##plt.savefig('phi_25.pdf')
#plt.show()
#sys.exit()
