# coding: utf-8
from dolfin import *
from numpy import array
import networkx as nx

def facet_neighborhood(mesh_):
    """Returns a dictionnary containing as key the index of the facets and as values the list of indices of the cells (or cell) containing the facet. """
    indices = dict([])

    for f in facets(mesh_):
        voisins_num = []
        for c in cells(f):
            voisins_num.append(c.index())

        indices[f.index()] = voisins_num
    return indices

#Adapt that to use the graph to store information
def connectivity_graph(problem):
    G = nx.Graph()

    #useful in the following
    dofmap_DG = problem.U_DG.dofmap()
    dofmap_CR = problem.U_CR.dofmap()

    #importing cell dofs
    for c in cells(problem.mesh): #Importing cells
        #Get the position of the barycentre
        bary = dofmap_DG.cell_dofs(c.index())
        #Get the num of the dofs in global DEM vector
        num_global_dof = dofmap_DG.entity_dofs(problem.mesh, problem.dim, array([c.index()], dtype="uintp"))
        
        #adding node to the graph
        G.add_node(c.index(), dof=num_global_dof, barycentre=bary)
        
    #importing connectivity and facet dofs
    for f in facets(problem.mesh):
        bnd = f.exterior() #cell on boundary or not
        mp = f.midpoint()
        
        #Getting number of neighbouring cells
        aux_bis = []
        for c in cells(f):
            aux_bis.append(c.index())

        #Get the num of the dofs in global DEM vector
        num_global_ddl_facet = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, array([f.index()], dtype="uintp")) #number of the dofs in CR
        #Get the position of the barycentre
        if problem.dim == 2:
            bary = array([mp.x(), mp.y()])
        elif problem.dim == 3:
            bary = array([mp.x(), mp.y(), mp.z()])
        else:
            raise ValueError('Problem with dimension of mesh')

        #adding the facets to the graph
        if not bnd: #add the link between two cell dofs       
            G.add_edge(aux_bis[0],aux_bis[1], num=num_global_ddl_facet[0] // problem.d, dof_CR=num_global_ddl_facet, barycentre=bary)
            
        elif bnd: #add the link between a cell dof and a boundary facet
            #number of the dof is total number of cells + num of the facet
            G.add_node(problem.nb_dof_DEM // problem.d + num_global_ddl_facet[0] // problem.d)
            G.add_edge(aux_bis[0], problem.nb_dof_DEM // problem.d + num_global_ddl_facet[0] // problem.d, num=num_global_ddl_facet[0] // problem.d, dof_CR=num_global_ddl_facet, barycentre=bary)
                
    return G
