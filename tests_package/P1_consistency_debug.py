# coding: utf-8
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import DEM_interpolation,local_project

#import pytest #for unit tests

#Size of mesh and number of elements
L = 0.5
nb_elt = 3

def test_reconstruction(mesh):
    h = mesh.hmax()
    dim = mesh.geometric_dimension()

    #DEM problem creation with reconstruction matrices
    problem = DEMProblem(mesh)

    #Testing P1 consistency and that's all
    x = SpatialCoordinate(mesh) #for disp
    if dim == 2:
        func = as_vector((x[0],x[1],x[0]))
    elif dim == 3:
        func = as_vector((x[0],x[1],x[2],x[0],x[1],x[2]))
    u,phi,tot = DEM_interpolation(func, problem)
    assert abs(max(u) - L) < h
    assert abs(min(u) + L) < h
    assert abs(min(phi) + L) < h
    assert abs(max(phi) - L) < h

    #CR interpolation
    test_CR = Function(problem.V_CR)
    reco_CR = problem.DEM_to_CR * tot
    test_CR.vector().set_local(reco_CR)
    assert round(max(reco_CR), 15) == L
    assert round(min(reco_CR), 15) == -L

    #Test on gradient on displacements
    test_CR_u,test_CR_phi = test_CR.split()
    if dim == 2:
        W = TensorFunctionSpace(mesh, 'DG', 0)
        W_PHI = VectorFunctionSpace(mesh, 'DG', 0)
    elif dim == 3:
        W = TensorFunctionSpace(mesh, 'DG', 0)
        W_PHI = W
    gradient_u = local_project(grad(test_CR_u), W)
    gradient_u_vec = gradient_u.vector().get_local()
    gradient_u_vec  = gradient_u_vec.reshape((problem.U_DG.dim() // dim,dim,dim))
    gradient_phi = local_project(grad(test_CR_phi), W_PHI)
    gradient_phi_vec = gradient_phi.vector().get_local()
    if dim == 2:
        gradient_phi_vec  = gradient_phi_vec.reshape((problem.PHI_DG.dim(),dim))
    elif dim == 3:
        gradient_phi_vec = gradient_phi_vec.reshape((problem.PHI_DG.dim() // dim,dim,dim))

    #Tests on disp
    assert round(min(gradient_u_vec[:,0,0]),13) == 1. and round(max(gradient_u_vec[:,0,0]),13) == 1.
    assert round(min(gradient_u_vec[:,0,1]),13) == 0. and round(max(gradient_u_vec[:,0,1]),13) == 0.
    assert round(min(gradient_u_vec[:,1,0]),13) == 0. and round(max(gradient_u_vec[:,1,0]),13) == 0.
    assert round(min(gradient_u_vec[:,1,1]),13) == 1. and round(max(gradient_u_vec[:,1,1]),13) == 1.
    #More tests for 3d functions
    if dim == 3:
        assert round(min(gradient_u_vec[:,0,2]),13) == 0. and round(max(gradient_u_vec[:,0,2]),13) == 0.
        assert round(min(gradient_u_vec[:,2,0]),13) == 0. and round(max(gradient_u_vec[:,2,0]),13) == 0.
        assert round(min(gradient_u_vec[:,1,2]),13) == 0. and round(max(gradient_u_vec[:,1,2]),13) == 0.
        assert round(min(gradient_u_vec[:,2,1]),13) == 0. and round(max(gradient_u_vec[:,2,1]),13) == 0.
        assert round(min(gradient_u_vec[:,2,2]),13) == 1. and round(max(gradient_u_vec[:,2,2]),13) == 1.

    #Test on gradient of rotations
    if dim == 2:
        assert round(min(gradient_phi_vec[:,0]),13) == 1. and round(max(gradient_phi_vec[:,0]),13) == 1.
        assert round(min(gradient_phi_vec[:,1]),13) == 0. and round(max(gradient_phi_vec[:,1]),13) == 0.
    elif dim == 3:
        assert round(min(gradient_phi_vec[:,0,0]),13) == 1. and round(max(gradient_phi_vec[:,0,0]),13) == 1.
        assert round(min(gradient_phi_vec[:,1,1]),13) == 1. and round(max(gradient_phi_vec[:,1,1]),13) == 1.
        assert round(min(gradient_phi_vec[:,2,2]),13) == 1. and round(max(gradient_phi_vec[:,2,2]),13) == 1.
        assert round(min(gradient_phi_vec[:,0,1]),13) == 0. and round(max(gradient_phi_vec[:,0,1]),13) == 0.
        assert round(min(gradient_phi_vec[:,1,0]),13) == 0. and round(max(gradient_phi_vec[:,1,0]),13) == 0.
        assert round(min(gradient_phi_vec[:,0,2]),13) == 0. and round(max(gradient_phi_vec[:,0,2]),13) == 0.
        assert round(min(gradient_phi_vec[:,2,0]),13) == 0. and round(max(gradient_phi_vec[:,2,0]),13) == 0.
        assert round(min(gradient_phi_vec[:,2,1]),13) == 0. and round(max(gradient_phi_vec[:,2,1]),13) == 0.
        assert round(min(gradient_phi_vec[:,1,2]),13) == 0. and round(max(gradient_phi_vec[:,1,2]),13) == 0.
        

    #Outputfile
    #file = File('P1_consistency.pvd')
    #file.write(test_CR)
    #file.write(gradient)

#mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
mesh = BoxMesh(Point(-L, -L, -L), Point(L, L, L), nb_elt, nb_elt, nb_elt)
test_reconstruction(mesh)
    #Outputfile
    #file.write(test_DG_1)
    #file.write(gradient_DG)
