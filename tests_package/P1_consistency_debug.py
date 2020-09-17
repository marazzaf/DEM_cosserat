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
    test_CR_u = Function(problem.U_CR)
    reco_CR_u = problem.DEM_to_CR * tot
    reco_CR_u = reco_CR_u.reshape((problem.V_DG.dim() // 3, 3))
    disp_CR_u = reco_CR_u[:,:2].flatten()
    test_CR_u.vector().set_local(reco_CR_u)
    gradient = local_project(grad(test_CR), W)
    gradient_vec = gradient.vector().get_local()
    gradient_vec  = gradient_vec.reshape((problem.V_DG.dim() // dim,dim,dim))
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

    #Test on gradient of rotations
        

    #Outputfile
    #file = File('P1_consistency.pvd')
    #file.write(test_CR)
    #file.write(gradient)

mesh = RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed")
test_reconstruction(mesh)
    #Outputfile
    #file.write(test_DG_1)
    #file.write(gradient_DG)
