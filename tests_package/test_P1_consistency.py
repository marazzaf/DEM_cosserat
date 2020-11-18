# coding: utf-8
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
from DEM_cosserat.miscellaneous import DEM_interpolation,local_project
import pytest #for unit tests
import matplotlib.pyplot as plt

#Size of mesh and number of elements
L = 0.5
nb_elt = 3

@pytest.mark.parametrize("mesh", [RectangleMesh(Point(-L,-L),Point(L,L),nb_elt,nb_elt,"crossed"), BoxMesh(Point(-L, -L, -L), Point(L, L, L), nb_elt, nb_elt, nb_elt)])
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

    ###test
    ##func_DG0 = local_project(func,problem.V_DG)
    ##func_DG1 = local_project(func_DG0, problem.V_DG1)
    ##print(func_DG1.vector().get_local())
    #vol = CellVolume(problem.mesh)
    #test_DG1 = TrialFunction(problem.V_DG1)
    #trial_DG0 = TestFunction(problem.V_DG)
    #mat = problem.d * inner(trial_DG0, test_DG1) / vol * dx
    #Mat = assemble(mat)
    #row,col,val = as_backend_type(Mat).mat().getValuesCSR()
    #Mat = csr_matrix((val, col, row))
    #print(Mat)
    #sys.exit()
    

    #test DG1 reconstruction
    #func = Constant(['1.'] * problem.d )
    #if problem.dim == 3:
    #    func = as_vector((0,0,0,0,0,x[2]))
    u,phi,tot = DEM_interpolation(func, problem)
    test_DG1 = Function(problem.V_DG1)
    reco_DG1 = problem.DEM_to_DG1 * tot
    test_DG1.vector().set_local(reco_DG1)
    #if problem.dim == 2:
    #    img = plot(test_DG1[2])
    #    plt.colorbar(img)
    #    plt.show()
    #assert round(max(reco_DG1), 14) == 1
    #assert round(min(reco_DG1), 14) == 1
    #print(reco_DG1[reco_DG1 > 1])
    #print(reco_DG1[reco_DG1 < 1])
    #print(reco_DG1[reco_DG1 > L])
    #print(reco_DG1[reco_DG1 < -L])
    assert round(max(reco_DG1), 14) == L
    assert round(min(reco_DG1), 14) == -L
    #sys.exit()

    #Test on gradient on displacements
    test_DG1_u,test_DG1_phi = test_DG1.split()
    gradient_u = local_project(grad(test_DG1_u), W)
    gradient_u_vec = gradient_u.vector().get_local()
    gradient_u_vec  = gradient_u_vec.reshape((problem.U_DG.dim() // dim,dim,dim))
    gradient_phi = local_project(grad(test_DG1_phi), W_PHI)
    gradient_phi_vec = gradient_phi.vector().get_local()
    if dim == 2:
        gradient_phi_vec  = gradient_phi_vec.reshape((problem.PHI_DG.dim(),dim))
    elif dim == 3:
        gradient_phi_vec = gradient_phi_vec.reshape((problem.PHI_DG.dim() // dim,dim,dim))

    #Tests on disp
    assert round(min(gradient_u_vec[:,0,0]),12) == 1. and round(max(gradient_u_vec[:,0,0]),12) == 1.
    assert round(min(gradient_u_vec[:,0,1]),12) == 0. and round(max(gradient_u_vec[:,0,1]),12) == 0.
    assert round(min(gradient_u_vec[:,1,0]),12) == 0. and round(max(gradient_u_vec[:,1,0]),12) == 0.
    assert round(min(gradient_u_vec[:,1,1]),12) == 1. and round(max(gradient_u_vec[:,1,1]),12) == 1.

    #More tests for 3d functions
    if dim == 3:
        assert round(min(gradient_u_vec[:,0,2]),12) == 0. and round(max(gradient_u_vec[:,0,2]),12) == 0.
        assert round(min(gradient_u_vec[:,2,0]),12) == 0. and round(max(gradient_u_vec[:,2,0]),12) == 0.
        assert round(min(gradient_u_vec[:,1,2]),12) == 0. and round(max(gradient_u_vec[:,1,2]),12) == 0.
        assert round(min(gradient_u_vec[:,2,1]),12) == 0. and round(max(gradient_u_vec[:,2,1]),12) == 0.
        assert round(min(gradient_u_vec[:,2,2]),12) == 1. and round(max(gradient_u_vec[:,2,2]),12) == 1.

    #Test on gradient of rotations
    if dim == 2:
        assert round(min(gradient_phi_vec[:,0]),12) == 1. and round(max(gradient_phi_vec[:,0]),12) == 1.
        assert round(min(gradient_phi_vec[:,1]),12) == 0. and round(max(gradient_phi_vec[:,1]),12) == 0.
    elif dim == 3:
        assert round(min(gradient_phi_vec[:,0,0]),12) == 1. and round(max(gradient_phi_vec[:,0,0]),12) == 1.
        assert round(min(gradient_phi_vec[:,1,1]),12) == 1. and round(max(gradient_phi_vec[:,1,1]),12) == 1.
        assert round(min(gradient_phi_vec[:,2,2]),12) == 1. and round(max(gradient_phi_vec[:,2,2]),12) == 1.
        assert round(min(gradient_phi_vec[:,0,1]),12) == 0. and round(max(gradient_phi_vec[:,0,1]),12) == 0.
        assert round(min(gradient_phi_vec[:,1,0]),12) == 0. and round(max(gradient_phi_vec[:,1,0]),12) == 0.
        assert round(min(gradient_phi_vec[:,0,2]),12) == 0. and round(max(gradient_phi_vec[:,0,2]),12) == 0.
        assert round(min(gradient_phi_vec[:,2,0]),12) == 0. and round(max(gradient_phi_vec[:,2,0]),12) == 0.
        assert round(min(gradient_phi_vec[:,2,1]),12) == 0. and round(max(gradient_phi_vec[:,2,1]),12) == 0.
        assert round(min(gradient_phi_vec[:,1,2]),12) == 0. and round(max(gradient_phi_vec[:,1,2]),12) == 0.
    
