#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
#import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import *
import pytest #for unit tests

# Mesh
L = 0.12
nb_elt = 10

# Parameters
nu = 0.25 # Poisson's ratio
G = 1e3 #Second lamé coefficient
E = 2*(1+nu)*G #Young Modulus
l = 0.1 # intrinsic length scale
a = 0.5 #last param in law
pen = 1e2 #penalty parameter

@pytest.mark.parametrize("mesh", [RectangleMesh(Point(-L,-L/2),Point(L,L/2),nb_elt,nb_elt,"crossed")])
def test_patch1(mesh):

    #Creating the DEM problem
    problem = DEMProblem(mesh, pen)
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

    #compliance tensor
    problem.micropolar_constants(E, nu, l, a)

    # Variational problem
    A = problem.elastic_bilinear_form()

    #Penalty matrix
    A += inner_penalty(problem) #_light

    #BC
    u_D = Expression('1e-3*(x[0]+0.5*x[1])', degree=1)
    v_D = Expression('1e-3*(x[0]+x[1])', degree=1)
    phi_D = Constant(0.25e-3)
    bc = [[0, u_D], [1, v_D], [2, phi_D]]
    #Assembling
    A += lhs_bnd_penalty(problem, boundary_parts, bc)
    rhs = rhs_bnd_penalty(problem, boundary_parts, bc)

    #Solving linear problem
    v_DG = Function(problem.V_DG)
    solve(PETScMatrix(A), v_DG.vector(), PETScVector(rhs), 'mumps')
    v_h = Function(problem.V_DG1)
    v_h.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
    u_h, phi_h = v_h.split()

    #Test
    sys.exit()
    F = FacetArea(problem.mesh)
    W = FunctionSpace(problem.mesh, 'CR', 1)
    w = TestFunction(W)
    aire = sum(assemble(w * ds).get_local())
    assert round(assemble(u_h[0] / aire * ds), 2) == 0
    assert round(assemble(u_h[1] / aire * ds), 2) == 0
    assert round(assemble(phi_h / aire * ds), 2) == 0

