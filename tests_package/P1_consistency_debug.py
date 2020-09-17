# coding: utf-8
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import DEM_interpolation,local_project

#import pytest #for unit tests
eps = 2e-15 #constant to compare floats. Possible to find a Python constant with that value ?
eps_2 = 1e-12 #constant to compare gradients to zero

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
    #rot = x[0]
    func = as_vector((x[0],x[1],x[0]))
    u,phi = DEM_interpolation(func, problem)
    assert abs(max(u) - L) < h
    assert abs(min(u) + L) < h
    assert abs(min(phi) + L) < h
    assert abs(max(phi) - L) < h

    #Functional Spaces
    U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
    U_DG_1 = VectorFunctionSpace(mesh, 'DG', 1)
    U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
    W = TensorFunctionSpace(mesh, 'DG', 0)

    #CR interpolation
    test_CR = Function(U_CR)
    reco_CR = problem.DEM_to_CR * u
    test_CR.vector().set_local(reco_CR)
    assert round(max(reco_CR), 13) == L
    assert round(max(reo_CR), 13) == L

    #Test on gradient
    gradient = local_project(grad(test_CR), W)
    gradient_vec = gradient.vector().get_local()
    gradient_vec  = gradient_vec.reshape((U_DG.dim() // d,d,dim))
    assert abs(min(gradient_vec[:,0,0]) - 1.) < eps_2 and abs(max(gradient_vec[:,0,0]) - 1.) < eps_2
    assert abs(min(gradient_vec[:,0,1])) < eps_2 and abs(max(gradient_vec[:,0,1])) < eps_2
    assert abs(min(gradient_vec[:,1,0])) < eps_2 and abs(max(gradient_vec[:,1,0])) < eps_2
    assert abs(min(gradient_vec[:,1,1]) - 1.) < eps_2 and abs(max(gradient_vec[:,1,1]) - 1.) < eps_2
    #More tests for 3d functions
    if d == 3:
        assert abs(min(gradient_vec[:,0,2])) < eps_2 and abs(max(gradient_vec[:,0,2])) < eps_2
        assert abs(min(gradient_vec[:,2,0])) < eps_2 and abs(max(gradient_vec[:,2,0])) < eps_2
        assert abs(min(gradient_vec[:,1,2])) < eps_2 and abs(max(gradient_vec[:,1,2])) < eps_2
        assert abs(min(gradient_vec[:,2,1])) < eps_2 and abs(max(gradient_vec[:,2,1])) < eps_2
        assert abs(min(gradient_vec[:,2,2]) - 1.) < eps_2 and abs(max(gradient_vec[:,2,2]) - 1.) < eps_2
        

    #Outputfile
    #file = File('P1_consistency.pvd')
    #file.write(test_CR)
    #file.write(gradient)

    #P1-discontinuous reconstruction
    test_DG_1 = Function(U_DG_1)
    test_DG_1.vector().set_local(problem.DEM_to_DG_1 * u)
    assert abs(max(test_DG_1.vector().get_local()) - L) < eps
    assert abs(min(test_DG_1.vector().get_local()) + L) < eps

    #Test on gradient
    gradient_DG = local_project(grad(test_DG_1), W)
    gradient_vec = gradient_DG.vector().get_local()
    gradient_vec  = gradient_vec.reshape((U_DG.dim() // d,d,dim))
    assert abs(min(gradient_vec[:,0,0]) - 1.) < eps_2 and abs(max(gradient_vec[:,0,0]) - 1.) < eps_2
    assert abs(min(gradient_vec[:,0,1])) < eps_2 and abs(max(gradient_vec[:,0,1])) < eps_2
    assert abs(min(gradient_vec[:,1,0])) < eps_2 and abs(max(gradient_vec[:,1,0])) < eps_2
    assert abs(min(gradient_vec[:,1,1]) - 1.) < eps_2 and abs(max(gradient_vec[:,1,1]) - 1.) < eps_2
    #More tests for 3d functions
    if d == 3:
        assert abs(min(gradient_vec[:,0,2])) < eps_2 and abs(max(gradient_vec[:,0,2])) < eps_2
        assert abs(min(gradient_vec[:,2,0])) < eps_2 and abs(max(gradient_vec[:,2,0])) < eps_2
        assert abs(min(gradient_vec[:,1,2])) < eps_2 and abs(max(gradient_vec[:,1,2])) < eps_2
        assert abs(min(gradient_vec[:,2,1])) < eps_2 and abs(max(gradient_vec[:,2,1])) < eps_2
        assert abs(min(gradient_vec[:,2,2]) - 1.) < eps_2 and abs(max(gradient_vec[:,2,2]) - 1.) < eps_2

mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
test_reconstruction(mesh)
    #Outputfile
    #file.write(test_DG_1)
    #file.write(gradient_DG)
