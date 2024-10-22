# coding: utf-8
from dolfin import *
import numpy as np
import networkx as nx

#Use the graph to store information
def connectivity_graph(problem):
    G = nx.Graph()

    #useful in the following
    dofmap_DG = problem.U_DG.dofmap()
    dofmap_DG_phi = problem.PHI_DG.dofmap()
    elt_DG = problem.U_DG.element()
    dofmap_CR = problem.U_CR.dofmap()
    dofmap_CR_phi = problem.PHI_CR.dofmap()

    #To count Dirichlet dofs
    count_Dirichlet = problem.nb_dof_DEM

    #importing cell dofs
    for c in cells(problem.mesh): #Importing cells
        #Get the position of the barycentre
        bary = elt_DG.tabulate_dof_coordinates(c)[0]
        #Get the num of the dofs in global DEM vector
        num_global_dof = dofmap_DG.entity_dofs(problem.mesh, problem.dim, np.array([c.index()], dtype="uintp"))
        num_global_dof_phi = dofmap_DG_phi.entity_dofs(problem.mesh, problem.dim, np.array([c.index()], dtype="uintp"))
        
        #adding node to the graph
        G.add_node(c.index(), dof_u=num_global_dof, barycentre=bary, dof_phi=num_global_dof_phi)
        
    #importing connectivity and facet dofs
    for f in facets(problem.mesh):
        bnd = f.exterior() #cell on boundary or not
        mp = f.midpoint()
        
        #Getting number of neighbouring cells
        aux_bis = []
        for c in cells(f):
            aux_bis.append(c.index())

        #Get the num of the dofs in global DEM vector
        num_global_dof_facet = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, np.array([f.index()], dtype="uintp")) #number of the dofs in CR
        num_global_dof_facet_phi = dofmap_CR_phi.entity_dofs(problem.mesh, problem.dim - 1, np.array([f.index()], dtype="uintp")) #number of the dofs in CR
        
        #Get the position of the barycentre
        if problem.dim == 2:
            bary = np.array([mp.x(), mp.y()])
        elif problem.dim == 3:
            bary = np.array([mp.x(), mp.y(), mp.z()])
        else:
            raise ValueError('Problem with dimension of mesh')
        
        #adding the facets to the graph
        if not bnd: #add the link between two cell dofs
            G.add_edge(aux_bis[0],aux_bis[1], num=f.index(), dof_CR_u=num_global_dof_facet, barycentre=bary, dof_CR_phi=num_global_dof_facet_phi, bnd=bnd)
            
        elif bnd: #add the link between a cell dof and a boundary facet
            #number of the dof is total number of cells + num of the facet
            if len(set(num_global_dof_facet) & problem.Dirichlet_CR_dofs) > 0:
                num_global_dof = list(np.arange(count_Dirichlet, count_Dirichlet+problem.d_u))
                num_global_dof_phi = list(np.arange(count_Dirichlet+problem.d_u,count_Dirichlet+problem.d_u+problem.d_phi))
                count_Dirichlet += problem.d
                G.add_node(problem.nb_dof_DEM + f.index(), dof_u=num_global_dof, dof_phi=num_global_dof_phi)
            else:
                G.add_node(problem.nb_dof_DEM + f.index())
            G.add_edge(aux_bis[0], problem.nb_dof_DEM + f.index(), num=f.index(), dof_CR_u=num_global_dof_facet, dof_CR_phi=num_global_dof_facet_phi, barycentre=bary, bnd=bnd)

    assert(problem.nb_dof_DEM + len(problem.Dirichlet_CR_dofs) == count_Dirichlet) #Checks that dof numbers are good on boundary
                
    return G
