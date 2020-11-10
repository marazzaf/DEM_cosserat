# coding: utf-8

from dolfin import *
from scipy.sparse import csr_matrix,dok_matrix
import numpy as np
from DEM_cosserat.reconstructions import *
from DEM_cosserat.mesh_related import *
from DEM_cosserat.miscellaneous import gradient_matrix

class DEMProblem:
    """ Class that will contain the basics of a DEM problem from the mesh and the dimension of the problem to reconstrucion matrices and gradient matrix."""
    def __init__(self, mesh, penalty_u=1., penalty_phi=1.):
        self.mesh = mesh
        self.dim = self.mesh.geometric_dimension()
        self.penalty_u = penalty_u
        self.penalty_phi = penalty_phi

        #Rotation is a scalar in 3d and a vector in 3d
        U_DG = VectorElement('DG', self.mesh.ufl_cell(), 0)
        U_DG1 = VectorElement('DG', self.mesh.ufl_cell(), 1)
        U_CR = VectorElement('CR', self.mesh.ufl_cell(), 1)
        WW = TensorElement('DG', self.mesh.ufl_cell(), 0)
        if self.dim == 2:
            PHI_DG = FiniteElement('DG', self.mesh.ufl_cell(), 0)
            PHI_DG1 = FiniteElement('DG', self.mesh.ufl_cell(), 1)
            PHI_CR = FiniteElement('CR', self.mesh.ufl_cell(), 1)
            PHI_W = VectorElement('DG', self.mesh.ufl_cell(), 0)
        elif self.dim == 3:
            PHI_DG = VectorElement('DG', self.mesh.ufl_cell(), 0)
            PHI_DG1 = VectorElement('DG', self.mesh.ufl_cell(), 1)
            PHI_CR = VectorElement('CR', self.mesh.ufl_cell(), 1)
            PHI_W = TensorElement('DG', self.mesh.ufl_cell(), 0)
        else:
            raise ValueError('Problem is whether 2d or 3d')
        #Spaces for DEM dofs
        self.V_DG = FunctionSpace(self.mesh, MixedElement(U_DG,PHI_DG))
        self.U_DG,self.PHI_DG = self.V_DG.split()
        self.nb_dof_DEM = self.V_DG.dofmap().global_dimension()

        #Dimension of Approximation space
        u = Function(self.V_DG)
        self.d = len(u)

        #Spaces for facet interpolations
        self.V_CR = FunctionSpace(self.mesh, MixedElement(U_CR,PHI_CR))
        self.U_CR,self.PHI_CR = self.V_CR.split()
        self.nb_dof_CR = self.V_CR.dofmap().global_dimension()
        
        #Spaces for gradients (strains and stresses)
        self.W = FunctionSpace(self.mesh, MixedElement(WW,PHI_W))
        self.WW,self.PHI_W = self.W.split()
        self.nb_dof_grad = self.W.dofmap().global_dimension()

        #Spaces for penalties
        self.V_DG1 = FunctionSpace(self.mesh, MixedElement(U_DG1,PHI_DG1))
        self.U_DG1,self.PHI_DG1 = self.V_DG1.split()
        self.nb_dof_DG1 = self.V_DG1.dofmap().global_dimension()
        
        #Creating the graph associated with the mesh
        self.Graph = connectivity_graph(self)

        #Computation of gradient matrix for inner penalty term
        self.mat_grad = gradient_matrix(self)

        #DEM reconstructions
        self.DEM_to_CR = DEM_to_CR_matrix(self)
        self.DEM_to_DG1 = DEM_to_DG1_matrix(self)

    #Importing useful functions
    from DEM_cosserat.miscellaneous import assemble_volume_load

    #Defining methods
    def D_Matrix(self, G, nu, N, l):
        self.G = G
        self.l = l
        a = 2*(1-nu)/(1-2*nu)
        b = 2*nu/(1-2*nu)
        c = 1/(1-N*N)
        d = (1-2*N*N)/(1-N*N)
        return G * as_matrix([[a,b,0,0], [b,a,0,0], [0,0,c,d], [0,0,d,c]]) #Not correct?
        #return self.G * as_matrix([[a,0,0,b], [0,c,d,0], [0,d,c,0], [b,0,0,a]]) #correct
        
    def strains(self, v, eta):
        gamma = as_vector([v[0].dx(0), v[1].dx(1), v[1].dx(0) - eta, v[0].dx(1) + eta])
        kappa = grad(eta)
        return gamma, kappa

    def stresses(self, strains):
        gamma,kappa = strains
        sigma = dot(self.D, gamma)
        mu = 4*self.G*self.l*self.l * kappa
        return sigma, mu


    def elastic_bilinear_form(self): #, strain, stress):
        u_CR,psi_CR = TrialFunctions(self.V_CR)
        v_CR,eta_CR = TestFunctions(self.V_CR)

        #Variationnal formulation
        def_test = self.strains(v_CR,eta_CR)
        def_trial = self.strains(u_CR, psi_CR)
        stress_trial = self.stresses(def_trial)
        a = (inner(def_test[0],stress_trial[0]) + inner(def_test[1],stress_trial[1])) * dx

        A = assemble(a)
        row,col,val = as_backend_type(A).mat().getValuesCSR()
        A = csr_matrix((val, col, row))
        return self.DEM_to_CR.T * A * self.DEM_to_CR

def inner_penalty(problem):
    """Creates the penalty matrix on inner facets to stabilize the DEM."""
    
    #assembling penalty factor
    h = CellDiameter(problem.mesh)
    h_avg = 0.5 * (h('+') + h('-'))

    #Writing penalty bilinear form
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)
    a_pen = problem.penalty_u / h_avg * inner(jump(u), jump(v)) * dS + problem.penalty_phi / h_avg * inner(jump(phi), jump(psi)) * dS

    #Assembling matrix
    A = assemble(a_pen)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    return problem.DEM_to_DG1.T * A * problem.DEM_to_DG1

def inner_penalty_light(problem):
    """Creates the penalty matrix on inner facets to stabilize the DEM."""
    h = CellDiameter(problem.mesh)
    h_avg = 0.5 * (h('+') + h('-'))

    #Average facet jump bilinear form
    u_DG,phi_DG = TrialFunctions(problem.V_DG1)
    v_CR,psi_CR = TestFunctions(problem.V_CR)
    F = FacetArea(problem.mesh)
    a_jump = sqrt(problem.penalty_u / h_avg) * inner(jump(u_DG), v_CR('+')) * dS + sqrt(problem.penalty_phi / h_avg) * inner(jump(phi_DG), psi_CR('+')) * dS
    A = assemble(a_jump)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    return problem.DEM_to_DG1.T * A.T * A * problem.DEM_to_DG1

#def penalties(problem):
#    """Creates the penalty matrix to stabilize the DEM."""
#
#    if problem.d == problem.dim:
#        tens_DG_0 = TensorFunctionSpace(problem.mesh, 'DG', 0)
#    elif problem.d == 1:
#        tens_DG_0 = VectorFunctionSpace(problem.mesh, 'DG', 0)
#    else:
#        raise ValueError
#        
#    dofmap_CR = problem.CR.dofmap()
#    elt_DG = problem.DG_0.element()
#    nb_ddl_CR = dofmap_CR.global_dimension()
#    dofmap_tens_DG_0 = tens_DG_0.dofmap()
#    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()
#
#    #assembling penalty factor
#    h = CellDiameter(problem.mesh)
#    h_avg = 0.5 * (h('+') + h('-'))
#    Aux = FunctionSpace(problem.mesh, 'CR', 1)
#    func = TestFunction(Aux)
#    pen_u = problem.penalty_u / h_avg * func('+') * dS + problem.penalty_u / h * func * ds
#    pen_phi = problem.penalty_phi / h_avg * func('+') * dS + problem.penalty_phi / h * func * ds
#    mat_u = assemble(pen_u).get_local()
#    mat_phi = assemble(pen_phi).get_local()
#
#    #creating jump matrix
#    mat_jump_1 = dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM))
#    mat_jump_2 = dok_matrix((problem.nb_dof_CR,problem.nb_dof_grad))
#    for f in facets(problem.mesh):
#        num_global_face = f.index()
#        num_global_ddl = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, np.array([num_global_face], dtype="uintp"))
#        coeff_pen = mat[num_global_ddl][0]
#        pos_bary_facet = problem.bary_facets[f.index()] #position barycentre of facet
#        
#        if len(problem.facet_num.get(f.index())) == 2: #Internal facet
#            for c,num_cell,sign in zip(cells(f),problem.facet_num.get(num_global_face),[1., -1.]):
#                #filling-out the DG 0 part of the jump
#                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,problem.d * num_cell : (num_cell+1) * problem.d] = sign*np.sqrt(coeff_pen)*np.eye(problem.d)
#                
#                #filling-out the DG 1 part of the jump...
#                pos_bary_cell = elt_DG.tabulate_dof_coordinates(c)[0]
#                diff = pos_bary_facet - pos_bary_cell
#                pen_diff = np.sqrt(coeff_pen)*diff
#                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
#                for num,dof_CR in enumerate(num_global_ddl):
#                    for i in range(problem.dim):
#                        mat_jump_2[dof_CR,tens_dof_position[(num % problem.d)*problem.d + i]] = sign*pen_diff[i]
#
#        elif len(problem.facet_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
#            #cell part
#            #filling-out the DG 0 part of the jump
#            num_cell = problem.facet_num.get(f.index())[0]
#            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,problem.d * num_cell : (num_cell+1) * problem.d] = np.sqrt(coeff_pen)*np.eye(problem.d)
#
#            #filling-out the DG 1 part of the jump
#            pos_bary_cell = problem.bary_cells.get(num_cell)
#            diff = pos_bary_facet - pos_bary_cell
#            pen_diff = np.sqrt(coeff_pen)*diff
#            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
#            for num,dof_CR in enumerate(num_global_ddl):
#                for i in range(problem.dim):
#                    mat_jump_2[dof_CR,tens_dof_position[(num % problem.d)*problem.d + i]] = pen_diff[i]
#
#            #boundary facet part
#            for v in vertices(f):
#                dof_vert = problem.num_ddl_vertex.get(v.index())
#                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,dof_vert[0]:dof_vert[-1]+1] = -np.sqrt(coeff_pen)*np.eye(problem.d) / problem.d
#            
#    mat_jump = mat_jump_1.tocsr() + mat_jump_2.tocsr() * problem.mat_grad * problem.DEM_to_CR
#    return mat_jump.T * mat_jump
