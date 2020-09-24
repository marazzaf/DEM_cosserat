# coding: utf-8
from scipy.sparse import dok_matrix,csr_matrix
from dolfin import *
import numpy as np
import networkx as nx
from itertools import combinations
from DEM_cosserat.mesh_related import *
import sys

def DEM_to_DG1_matrix(problem):
    matrice_resultat_1 = dok_matrix((problem.nb_dof_DG1,problem.nb_dof_DEM)) #Empty matrix
    matrice_resultat_2 = dok_matrix((problem.nb_dof_DG1,problem.nb_dof_grad)) #Empty matrix

    #Useful in the following
    dofmap_DG_0 = problem.V_DG.dofmap()
    dofmap_DG_1 = problem.V_DG1.dofmap()
    dofmap_tens_DG_0 = problem.W.dofmap()
    elt_1 = problem.V_DG1.element()

    for c in cells(problem.mesh):
        index_cell = c.index()
        dof_position = dofmap_DG_1.cell_dofs(index_cell)

        #filling out the matrix to have the constant cell value
        DG_0_dofs = dofmap_DG_0.cell_dofs(index_cell)
        for dof in dof_position:
            matrice_resultat_1[dof, DG_0_dofs[dof % problem.d]] = 1.
            
        #filling out part to add the gradient term
        position_barycentre = problem.Graph.nodes[index_cell]['barycentre']
        pos_dof_DG_1 = elt_1.tabulate_dof_coordinates(c)
        tens_dof_position = dofmap_tens_DG_0.cell_dofs(index_cell)
        for dof,pos in zip(dof_position,pos_dof_DG_1): #loop on quadrature points
            diff = pos - position_barycentre
            for i in range(problem.dim):
                #print(dof,(dof % problem.d) * problem.dim + i)
                #print(dof,tens_dof_position[(dof % problem.d) * problem.dim + i])
                matrice_resultat_2[dof, tens_dof_position[(dof % problem.d) * problem.dim + i]] = diff[i]
                #matrice_resultat_2[dof, (dof % problem.d) * problem.dim + i] = diff[i]
        #sys.exit()
    return matrice_resultat_1.tocsr() + matrice_resultat_2.tocsr() * problem.mat_grad * problem.DEM_to_CR

def facet_interpolation(problem):
    """Computes the reconstruction in the facets of the meh from the dofs of the DEM."""
    #assert isinstance(problem,DEMProblem)
    
    #To store the results of the computations
    res_num = dict([])
    res_coord = dict([])
    res_num_phi = dict([])

    #Loop over all facets of the mesh
    for c1,c2 in problem.Graph.edges():
        c1,c2 = min(c1,c2),max(c1,c2)
        facet = problem.Graph[c1][c2]
        num_facet = facet['num']
        x = facet['barycentre'] #Position of the barycentre of the facet

        #Defining the set of dofs in which to look for the convex for barycentric reconstruction
        if not facet['bnd']: #inner facet
            #Computing the neighbours
            if problem.dim == 2:
                path_1 = nx.neighbors(problem.Graph, c1)
                path_2 = nx.neighbors(problem.Graph, c2)
            elif problem.dim == 3:
                path_1 = nx.single_source_shortest_path(problem.Graph, c1, cutoff=2)
                path_2 = nx.single_source_shortest_path(problem.Graph, c2, cutoff=2)
            #Removing bnd neighbours of first cell
            path_1 = np.array(list(path_1))
            for_deletion = np.where(path_1 >= problem.nb_dof_DEM)
            path_1[for_deletion] = -1
            path_1 = set(path_1) - {-1}
            #Removing bnd neighbours of second cell
            path_2 = np.array(list(path_2))
            for_deletion = np.where(path_2 >= problem.nb_dof_DEM)
            path_2[for_deletion] = -1
            path_2 = set(path_2) - {-1}

            #Final set of neighbours to choose reconstruction from
            neigh_pool = path_1 | path_2
            
        else: #boundary facet
            assert c2 >= problem.nb_dof_DEM #Check that cell_2 is a boundary node that is not useful

            neigh_pool = set(nx.single_source_shortest_path(problem.Graph, c1, cutoff=2)) - {num_facet + problem.nb_dof_DEM}

            neigh_pool = np.array(list(neigh_pool))
            for_deletion = np.where(neigh_pool >= problem.nb_dof_DEM)
            neigh_pool[for_deletion] = -1
            neigh_pool = set(neigh_pool) - {-1}

        #Empty sets to store result of search
        chosen_coord_bary = []
        coord_num = []
        coord_num_phi = []
        
        #Search of the simplex
        for dof_num in combinations(neigh_pool, problem.dim+1): #test reconstruction with a set of right size
            
            #Dof positions to assemble matrix to compute barycentric coordinates
            list_positions = []   
            for l in dof_num:
                list_positions.append(problem.Graph.nodes[l]['barycentre'])

            #Computation of barycentric coordinates
            A = np.array(list_positions)
            A = A[1:,:] - A[0,:]
            b = np.array(x - list_positions[0])
            try:
                partial_coord_bary = np.linalg.solve(A.T,b)
            except np.linalg.LinAlgError: #singular matrix
                pass
            else:
                coord_bary = np.append(1. - partial_coord_bary.sum(), partial_coord_bary)
                if max(abs(coord_bary)) < 10.:
                    chosen_coord_bary = coord_bary
                    for l in dof_num:
                        coord_num.append(problem.Graph.nodes[l]['dof_u'])
                        coord_num_phi.append(problem.Graph.nodes[l]['dof_phi'])

                    break #search is finished when we get a simplex that works
                
        #Tests if search was fruitful
        assert len(chosen_coord_bary) > 0 #otherwise no coordinates have been computed
        assert len(chosen_coord_bary) == len(coord_num) == len(coord_num_phi)

        res_num[num_facet] = coord_num
        res_num_phi[num_facet] = coord_num_phi
        res_coord[num_facet] = chosen_coord_bary
                                
    return res_num,res_num_phi,res_coord

def DEM_to_CR_matrix(problem):
    #assert isinstance(problem,DEMProblem)

    #dofmaps to fill the matrix
    dofmap_U_CR = problem.U_CR.dofmap()
    dofmap_PHI_CR = problem.PHI_CR.dofmap()
    
    #Computing the facet reconstructions
    simplex_num,simplex_num_phi,simplex_coord = facet_interpolation(problem)

    #Storing the facet reconstructions in a matrix
    result_matrix = dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM)) #Empty matrix
    for c1,c2 in problem.Graph.edges():
        num_global_facet = problem.Graph[c1][c2]['num']
        num_global_ddl = problem.Graph[c1][c2]['dof_CR_u']
        num_global_ddl_phi = problem.Graph[c1][c2]['dof_CR_phi']
        
        simplex_f = simplex_num.get(num_global_facet)
        simplex_phi = simplex_num_phi.get(num_global_facet)
        simplex_c = simplex_coord.get(num_global_facet)

        #Filling the reconstruction matrix
        for i,j,k in zip(simplex_f,simplex_c,simplex_phi):
            result_matrix[num_global_ddl[0],i[0]] = j #Disp x
            result_matrix[num_global_ddl[1],i[1]] = j #Disp y
            result_matrix[num_global_ddl_phi[0],k[0]] = j #Rotation
            if problem.dim == 3:
                result_matrix[num_global_ddl[2],i[2]] = j #Disp z
                result_matrix[num_global_ddl_phi[1],k[1]] = j #Rotation y
                result_matrix[num_global_ddl_phi[2],k[2]] = j #Rotation z
        
    return result_matrix.tocsr()

