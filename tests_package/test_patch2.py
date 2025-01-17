#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
import numpy as np
from DEM_cosserat.miscellaneous import *
import pytest #for unit tests
import matplotlib.pyplot as plt

# Mesh
L = 0.12
nb_elt = 100

# Parameters
nu = 0.25 # Poisson's ratio
G = 1e3 #Second lamé coefficient
E = 2*(1+nu)*G #Young Modulus
l = 0.1 # intrinsic length scale
a = 0.5 #last param in law
pen = 1 #penalty parameter

@pytest.mark.parametrize("mesh", [RectangleMesh(Point(-L,-L/2),Point(L,L/2),nb_elt,nb_elt,"crossed")])
def test_patch2(mesh):

    #Creating the DEM problem
    problem = DEMProblem(mesh, pen)
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

    #compliance tensor
    problem.micropolar_constants(E, nu, l, a)

    # Variational problem
    A = problem.elastic_bilinear_form()

    #Penalty matrix
    A += inner_penalty(problem)

    #Volume load
    q = Constant((0,0,1))
    rhs = problem.assemble_volume_load(q)

    #BC
    u_D = Expression('1e-3*(x[0]+0.5*x[1])', degree=1)
    v_D = Expression('1e-3*(x[0]+x[1])', degree=1)
    alpha = -2 #-2
    phi_D = Constant(0.25e-3*(1+alpha))
    bc = [[0, u_D], [1, v_D], [2, phi_D]]
    #Assembling
    A += lhs_bnd_penalty(problem, boundary_parts, bc)
    rhs += rhs_bnd_penalty(problem, boundary_parts, bc)

    #Solving linear problem
    v_DG = Function(problem.V_DG)
    solve(PETScMatrix(A), v_DG.vector(), PETScVector(rhs), 'mumps')
    v_h = Function(problem.V_DG1)
    v_h.vector()[:] = problem.DEM_to_DG1 * v_DG.vector().vec()
    u_h, phi_h = v_h.split()

    #Computing stresses
    strains = problem.strains_2d(u_h, phi_h)
    sigma,mu = problem.stresses_2d(strains)

    #Test
    W = FunctionSpace(problem.mesh, 'DG', 0)
    #Testing on all elements
    #Testing stresses
    sigma_00 = local_project(sigma[0,0], W).vector().get_local()
    assert (np.round(sigma_00, 1) == 4).all()
    sigma_11 = local_project(sigma[1,1], W).vector().get_local()
    assert (np.round(sigma_11, 1) == 4).all()
    sigma_10 = local_project(sigma[1,0], W).vector().get_local()
    assert (np.round(sigma_10, 0) == 2).all()
    sigma_01 = local_project(sigma[0,1], W).vector().get_local()
    assert (np.round(sigma_01, 0) == 1).all()
    #Testing moments
    mu_0 = local_project(mu[0], W).vector().get_local()
    assert (np.round(mu_0, 0)  == 0).all()
    mu_1 = local_project(mu[1], W).vector().get_local()
    assert (np.round(mu_1, 0)  == 0).all()

