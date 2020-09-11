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
    nb_dof_CR = 2*dofmap_CR.global_dimension() #modify that if I create a mixed space

    #useful auxiliary functions
    vol_c = CellVolume(mesh_) #Pour volume des particules voisines
    hF = FacetArea(mesh_)
    n = FacetNormal(mesh_)
    scalar_DG = FunctionSpace(mesh_, 'DG', 0) #for volumes
    f_DG = TestFunction(scalar_DG)
    scalar_CR = FunctionSpace(mesh_, 'CR', 1) #for surfaces
    f_CR = TestFunction(scalar_CR)
    vectorial_CR = VectorFunctionSpace(mesh_, 'CR', 1) #for normals
    v_CR = TestFunction(vectorial_CR)



    #importing cell dofs
    for c in cells(mesh_): #Importing cells
        #Get the position of the barycentre
        bary = dofmap_DG.cell_dofs(c.index())
        #Get the num of the dofs in global DEM vector
        num_global_dof = dofmap_DG.entity_dofs(problem.mesh, dim, np.array([c.index()], dtype="uintp"))
        
        #adding node to the graph
        G.add_node(c.index(), dof=num_global_dof, pos=bary)

    #CONTINUE HERE !!!!
        
    #importing connectivity and facet dofs
    for f in facets(mesh_):
        aux_bis = [] #number of the cells
        for c in cells(f):
            aux_bis.append(c.index())
        num_global_ddl_facet = dofmap_CR.entity_dofs(mesh_, dim - 1, np.array([f.index()], dtype="uintp")) #number of the dofs in CR
        #computing quantites related to the facets
        vert = []
        vert_ind = []
        for v in vertices(f):
            vert.append( np.array(v.point()[:])[:dim] )
            vert_ind.append(v.index())
        normal = normals[num_global_ddl_facet[0] // d_, :]
        area = areas[num_global_ddl_facet[0] // d_]
        #facet barycentre computation
        vert = np.array(vert)
        bary = vert.sum(axis=0) / vert.shape[0]
        #index of the edges of the facet
        Edges = set()
        if dim == 3:
            for e in edges(f):
                Edges.add(e.index())

        #adding the facets to the graph
        if len(aux_bis) == 2: #add the link between two cell dofs
            #putting normals in the order of lowest cell number towards biggest cell number
            n1 = min(aux_bis[0],aux_bis[1])
            bary_n1 = G.node[n1]['pos']
            n2 = max(aux_bis[0],aux_bis[1])
            bary_n2 = G.node[n2]['pos']
         
            #adding edge
            G.add_edge(aux_bis[0],aux_bis[1], num=num_global_ddl_facet[0] // d_, recon=set([]), dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, normal=normal, vertices=vert, edges=Edges, pen_factor=pen_factor[num_global_ddl_facet[0] // d_], breakable=True) #, vertices_ind=vert_ind)
            
        elif len(aux_bis) == 1: #add the link between a cell dof and a boundary facet dof
            for c in cells(f): #only one cell contains the boundary facet
                bary_cell = G.node[c.index()]['pos']
            #computation of volume associated to the facet for mass matrix
            if dim == 2:
                vol_facet = 0.5 * np.linalg.norm(np.cross(vert[0] - bary_cell, vert[1] - bary_cell))
            elif dim == 3:
                vol_facet = np.linalg.norm(np.dot( np.cross(vert[0] - bary_cell, vert[1] - bary_cell), vert[2] - bary_cell )) / 6.

            #checking if adding "dofs" for Dirichlet BC
            nb_dofs = len(dirichlet_dofs & set(num_global_ddl_facet))
            aux = list(np.arange(count, count+nb_dofs))
            count += nb_dofs
            components = sorted(list(dirichlet_dofs & set(num_global_ddl_facet)))
            components = np.array(components) % d_
            
            #number of the dof is total number of cells + num of the facet
            G.add_node(nb_ddl_cells // d_ + num_global_ddl_facet[0] // d_, pos=bary, dof=aux, dirichlet_components=components)
            G.add_edge(aux_bis[0], nb_ddl_cells // d_ + num_global_ddl_facet[0] // d_, num=num_global_ddl_facet[0] // d_, dof_CR=num_global_ddl_facet, measure=area, barycentre=bary, normal=normal, vertices=vert, pen_factor=pen_factor[num_global_ddl_facet[0] // d_], breakable=False)
            G.node[aux_bis[0]]['bnd'] = True #Cell is on the boundary of the domain
                
    return G