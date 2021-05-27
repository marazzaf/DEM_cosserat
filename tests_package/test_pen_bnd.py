#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
import pytest #for unit tests

# Mesh
L = 0.5
nb_elt = 10

# Parameters
nu = 0.3 # Poisson's ratio
E = 1 #Young Modulus
l = L/10 # intrinsic length scale
a = 0.5
pen = 1e2 #penalty parameter

@pytest.mark.parametrize("mesh", [RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")])
def test_pen_bnd(mesh):

    #Creating the DEM problem
    problem = DEMProblem(mesh, pen)
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

    #compliance tensor
    problem.micropolar_constants(E, nu, l, a)

    # Variational problem
    A = problem.elastic_bilinear_form()

    #Penalty matrix
    A += inner_penalty(problem)

    #rhs
    t = Constant((-10,-10,0))
    rhs = problem.assemble_volume_load(t)

    #Nitsche penalty bilinear form. Homogeneous Dirichlet in this case.
    #A += lhs_nitsche_penalty(problem, strain, stresses)
    bc = [[0, Constant(0)], [1, Constant(0)], [2, Constant(0)]]
    #bc = [[0, Constant(1)], [1, Constant(1)], [2, Constant(1)]]
    A += lhs_bnd_penalty(problem, boundary_parts, bc)
    rhs += rhs_bnd_penalty(problem, boundary_parts, bc)
    #A += lhs_bnd_penalty(problem)

    #Solving linear problem
    v_DG = Function(problem.V_DG)
    solve(PETScMatrix(A), v_DG.vector(), PETScVector(rhs), 'mumps')
    v_h = Function(problem.V_DG1)
    v_h.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
    u_h, phi_h = v_h.split()

    F = FacetArea(problem.mesh)
    W = FunctionSpace(problem.mesh, 'CR', 1)
    w = TestFunction(W)
    aire = sum(assemble(w * ds).get_local())
    assert round(assemble(u_h[0] / aire * ds), 1) == 0
    assert round(assemble(u_h[1] / aire * ds), 1) == 0
    assert round(assemble(phi_h / aire * ds), 1) == 0

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
    #plt.savefig('u_x_no_pen.pdf')
    #plt.show()
    #fig = plot(u_h[1])
    #plt.colorbar(fig)
    #plt.savefig('u_y_no_pen.pdf')
    #plt.show()
    #fig = plot(phi_h)
    #plt.colorbar(fig)
    #plt.savefig('phi_no_pen.pdf')
    #plt.show()
