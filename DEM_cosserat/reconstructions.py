# coding: utf-8
import scipy.sparse as sp
from dolfin import *
import numpy as np
import networkx as nx
from itertools import combinations
from DEM_cosserat.mesh_related import *
from DEM_cosserat.errors import *
#from DEM_cosserat.DEM import DEMProblem


#def DEM_to_DG_matrix(problem,nb_dof_ccG_): #still useful? Not sure
#    """Creates a csr companion matrix to get the cells values of a DEM vector."""
#    assert isinstance(problem,DEMProblem)
#    
#    nb_cell_dofs = problem.DG_0.dofmap().global_dimension()
#    return sp.eye(nb_cell_dofs, n = nb_dof_ccG_, format='csr')
#
#def DEM_to_DG_1_matrix(problem, nb_dof_ccG_, DEM_to_CR):
#    assert isinstance(problem,DEMProblem)
#    
#    EDG_0 = problem.DG_0
#    EDG_1 = problem.DG_1
#    tens_DG_0 = problem.W
#        
#    dofmap_DG_0 = EDG_0.dofmap()
#    dofmap_DG_1 = EDG_1.dofmap()
#    dofmap_tens_DG_0 = tens_DG_0.dofmap()
#    elt_0 = EDG_0.element()
#    elt_1 = EDG_1.element()
#    nb_total_dof_DG_1 = dofmap_DG_1.global_dimension()
#    nb_dof_grad = dofmap_tens_DG_0.global_dimension()
#    matrice_resultat_1 = sp.dok_matrix((nb_total_dof_DG_1,nb_dof_ccG_)) #Empty matrix
#    matrice_resultat_2 = sp.dok_matrix((nb_total_dof_DG_1,nb_dof_grad)) #Empty matrix
#    
#    for c in cells(problem.mesh):
#        index_cell = c.index()
#        dof_position = dofmap_DG_1.cell_dofs(index_cell)
#
#        #filling-in the matrix to have the constant cell value
#        DG_0_dofs = dofmap_DG_0.cell_dofs(index_cell)
#        for dof in dof_position:
#            matrice_resultat_1[dof, DG_0_dofs[dof % problem.d]] = 1.
#
#        #filling-in part to add the gradient term
#        position_barycentre = elt_0.tabulate_dof_coordinates(c)[0]
#        pos_dof_DG_1 = elt_1.tabulate_dof_coordinates(c)
#        tens_dof_position = dofmap_tens_DG_0.cell_dofs(index_cell)
#        for dof,pos in zip(dof_position,pos_dof_DG_1): #loop on quadrature points
#            diff = pos - position_barycentre
#            for i in range(problem.dim):
#                matrice_resultat_2[dof, tens_dof_position[(dof % problem.d)*problem.d + i]] = diff[i]
#        
#    return matrice_resultat_1.tocsr() +  matrice_resultat_2.tocsr() * problem.mat_grad * DEM_to_CR

#def gradient_matrix(problem):
#    """Creates a matrix computing the cell-wise gradient from the facet values stored in a Crouzeix-raviart FE vector."""
#    vol = CellVolume(problem.mesh)
#
#    #variational form gradient
#    u_CR = TrialFunction(problem.U_CR)
#    Dv_DG = TestFunction(problem.W)
#    a = inner(grad(u_CR), Dv_DG) / vol * dx
#    A = assemble(a)
#    row,col,val = as_backend_type(A).mat().getValuesCSR()
#    result = sp.csr_matrix((val, col, row))
#
#    if problem.dim == 3:
#        return result,result
#    elif problem.dim == 2:
#        u_CR = TrialFunction(problem.PHI_CR)
#        Dv_DG = TestFunction(problem.PHI_W)
#        a = inner(grad(u_CR), Dv_DG) / vol * dx
#        A = assemble(a)
#        row,col,val = as_backend_type(A).mat().getValuesCSR()
#        return result,sp.csr_matrix((val, col, row))
#    else:
#        raise ValueError('Wrong dimension')

def facet_interpolation(problem):
    """Computes the reconstruction in the facets of the meh from the dofs of the DEM."""
    #assert isinstance(problem,DEMProblem)

    #for c in cells(problem.mesh):
    #    print(problem.U_DG.element().tabulate_dof_coordinates(c))
    #    sys.exit()
    
    #To store the results of the computations
    res_num = dict([])
    res_coord = dict([])

    #Loop over all facets of the mesh
    for c1,c2 in problem.Graph.edges():
        c1,c2 = min(c1,c2),max(c1,c2)
        facet = problem.Graph[c1][c2]
        num_facet = facet['num']
        cell_1 = problem.Graph.node[c1]
        cell_2 = problem.Graph.node[c2]
        x = facet['barycentre'] #Position of the barycentre of the facet

        #Defining the set of dofs in which to look for the convex for barycentric reconstruction
        if not facet['bnd']: #inner facet
            #Neighbours of first cell
            path_1 = nx.neighbors(problem.Graph, c1)
            path_1 = np.array(list(path_1))
            for_deletion = np.where(np.absolute(path_1) >= problem.nb_dof_DEM // problem.d)
            path_1[for_deletion] = -1
            path_1 = set(path_1) - {-1}
            #Neighbours of second cell
            path_2 = nx.neighbors(problem.Graph, c2)
            path_2 = np.array(list(path_2))
            for_deletion = np.where(np.absolute(path_2) >= problem.nb_dof_DEM // problem.d)
            path_2[for_deletion] = -1
            path_2 = set(path_2) - {-1}
            neigh_pool = path_1 | path_2
            
        else: #boundary facet
            assert c2 >= problem.nb_dof_DEM // problem.d #Check that cell_2 is a boundary node that is not useful

            neigh_pool = set(nx.single_source_shortest_path(problem.Graph, c1, cutoff=2)) - {num_facet + problem.nb_dof_DEM // problem.d}

            neigh_pool = np.array(list(neigh_pool))
            for_deletion = np.where(np.absolute(neigh_pool) >= problem.nb_dof_DEM // problem.d)
            neigh_pool[for_deletion] = -1
            neigh_pool = set(neigh_pool) - {-1}

        #Finding the convex
        for dof_num in combinations(neigh_pool, problem.dim+1): #test reconstruction with a set of right size
            chosen_coord_bary = []
            coord_num = []
            
            #Dof positions to assemble matrix to compute barycentric coordinates
            list_positions = []   
            for l in dof_num:
                list_positions.append(problem.Graph.node[l]['barycentre'])

            #Computation of barycentric coordinates
            A = np.array(list_positions)
            A = A[1:,:] - A[0,:]
            b = np.array(x - list_positions[0])
            try:
                aux_coord_bary = np.linalg.solve(A.T,b)
            except np.linalg.LinAlgError: #singular matrix
                pass
            else:
                if max(max(abs(aux_coord_bary)),1.-aux_coord_bary.sum()) < 10.:
                    chosen_coord_bary = np.append(1. - aux_coord_bary.sum(), aux_coord_bary)
                    coord_num = []
                    for l in dof_num:
                        #assert len(G_.node[l]['dof']) > 0
                        coord_num.append(problem.Graph.node[l]['dof'])

            #Tests if search was fruitful
            assert len(chosen_coord_bary) > 0 #otherwise no coordinates have been computed
            #else:
            #    raise ConvexError('Not possible to find a non-degenerate simplex for the facet reconstruction.\n')
            res_num[f] = coord_num
            res_coord[f] = chosen_coord_bary
                                
    return res_num,res_coord

def DEM_to_CR_matrix(problem):
    #assert isinstance(problem,DEMProblem)

    #dofmaps to fill the matrix
    dofmap_U_CR = problem.U_CR.dofmap()
    dofmap_PHI_CR = problem.PHI_CR.dofmap()
    
    #Computing the facet reconstructions
    convex_num,convex_coord = facet_interpolation(problem)

    sys.exit()

    #Storing the facet reconstructions in a matrix
    result_matrix = sp.dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM)) #Empty matrix
    for f in facets(problem.mesh):
        num_global_facet = f.index()
        num_global_ddl = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, array([num_global_facet], dtype="uintp"))
        convexe_f = convex_num.get(num_global_facet)
        convexe_c = convex_coord.get(num_global_facet)

        for i,j in zip(convexe_f,convexe_c):
            result_matrix[num_global_ddl[0],i[0]] = j
            if problem.d >= 2:
                result_matrix[num_global_ddl[1],i[1]] = j
            if problem.d == 3:
                result_matrix[num_global_ddl[2],i[2]] = j
        
    return result_matrix.tocsr()

#def compute_all_reconstruction_matrices(problem):
#    """Computes all the required reconstruction matrices."""
#
#    #calling functions to construct the matrices
#    DEM_to_DG = DEM_to_DG_matrix(problem, problem.nb_dof_DEM)
#    DEM_to_CR = DEM_to_CR_matrix(problem, problem.nb_dof_DEM, problem.facet_num, problem.vertex_associe_face, problem.num_ddl_vertex, problem.pos_ddl_vertex)
#    DEM_to_DG_1 = DEM_to_DG_1_matrix(problem, problem.nb_dof_DEM, DEM_to_CR)
#
#    return DEM_to_DG, DEM_to_CG, DEM_to_CR, DEM_to_DG_1
