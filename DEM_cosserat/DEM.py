# coding: utf-8

from dolfin import *
from scipy.sparse import csr_matrix,dok_matrix
import numpy as np
from DEM_cosserat.errors import *
from DEM_cosserat.reconstructions import DEM_to_CR_matrix
from DEM_cosserat.mesh_related import *
from DEM_cosserat.miscellaneous import Dirichlet_BC,schur_matrices
import sys

class DEMProblem:
    """ Class that will contain the basics of a DEM problem from the mesh and the dimension of the problem to reconstrucion matrices and gradient matrix."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.dim = self.mesh.geometric_dimension()

        #Rotation is a scalar in 3d and a vector in 3d
        U_DG = VectorElement('DG', self.mesh.ufl_cell(), 0)
        U_CR = VectorElement('CR', self.mesh.ufl_cell(), 1)
        WW = TensorElement('DG', self.mesh.ufl_cell(), 0)
        if self.dim == 2:
            PHI_DG = FiniteElement('DG', self.mesh.ufl_cell(), 0)
            PHI_CR = FiniteElement('CR', self.mesh.ufl_cell(), 1)
            PHI_W = VectorElement('DG', self.mesh.ufl_cell(), 0)
        elif self.dim == 3:
            PHI_DG = VectorElement('DG', self.mesh.ufl_cell(), 0)
            PHI_CR = VectorElement('CR', self.mesh.ufl_cell(), 1)
            PHI_W = TensorElement('DG', self.mesh.ufl_cell(), 0)
        else:
            raise ValueError('Problem is whether scalar or vectorial')
        #Spaces for DEM dofs
        self.V_DG = FunctionSpace(self.mesh, MixedElement(U_DG,PHI_DG))
        self.U_DG,self.PHI_DG = self.V_DG.split()
        self.nb_dof_DEM = self.V_DG.dofmap().global_dimension()

        #Spaces for facet interpolations
        self.V_CR = FunctionSpace(self.mesh, MixedElement(U_CR,PHI_CR))
        self.U_CR,self.PHI_CR = self.V_CR.split()
        self.nb_dof_CR = self.V_CR.dofmap().global_dimension()
        #Spaces for gradients (strains and stresses)
        self.W = FunctionSpace(self.mesh, MixedElement(WW,PHI_W))
        self.WW,self.PHI_W = self.W.split()
        self.nb_dof_grad = self.W.dofmap().global_dimension()

        ##what to do with these ?
        #self.DG_1 = VectorFunctionSpace(self.mesh, 'DG', 1)

        ##gradient
        #self.mat_grad,self.mat_grad_phi = gradient_matrix(self)
        #Creating the graph associated with the mesh
        self.Graph = connectivity_graph(self)

        ##Still useful ? Not sure with the graph...
        #self.facet_num = facet_neighborhood(self.mesh)

        #DEM reconstructions
        #self.DEM_to_DG, self.DEM_to_CR, self.DEM_to_DG_1 = compute_all_reconstruction_matrices(self)
        #print('Reconstruction matrices ok!')
        self.DEM_to_CR = DEM_to_CR_matrix(self)

        #Dirichlet conditions

   #def for_dirichlet(self, A, boundary_dirichlet=None):
   #    hF = FacetArea(self.mesh)
   #    v_CG = TestFunction(self.CG)
   #    if boundary_dirichlet == None: #dependence on self.d ???
   #        form_dirichlet = inner(v_CG('+'),as_vector((1.,1.))) / hF * ds
   #    else:
   #        form_dirichlet = inner(v_CG('+'),as_vector((1.,1.))) / hF * ds(boundary_dirichlet)
   #    A_BC = Dirichlet_BC(form_dirichlet, self.DEM_to_CG)
   #    self.mat_not_D,self.mat_D = schur_matrices(A_BC)
   #    #A_D = mat_D * A * mat_D.T
   #    A_not_D = self.mat_not_D * A * self.mat_not_D.T
   #    B = self.mat_not_D * A * self.mat_D.T
   #    return A_not_D,B


def elastic_bilinear_form(problem, D, strain, stress):
    u_CR,psi_CR = TrialFunctions(problem.V_CR)
    v_CR,eta_CR = TestFunctions(problem.V_CR)

    #Variationnal formulation
    def_test = strain(v_CR,eta_CR)
    def_trial = strain(u_CR, psi_CR)
    stress_trial = stress(D,def_trial)
    a = (inner(def_test[0],stress_trial[0]) + inner(def_test[1],stress_trial[1])) * dx
    
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))
    return problem.DEM_to_CR.T * A * problem.DEM_to_CR

def penalties(problem):
    """Creates the penalty matrix to stabilize the DEM."""

    if problem.d == problem.dim:
        tens_DG_0 = TensorFunctionSpace(problem.mesh, 'DG', 0)
    elif problem.d == 1:
        tens_DG_0 = VectorFunctionSpace(problem.mesh, 'DG', 0)
    else:
        raise ValueError
        
    dofmap_CR = problem.CR.dofmap()
    elt_DG = problem.DG_0.element()
    nb_ddl_CR = dofmap_CR.global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

    #assembling penalty factor
    vol = CellVolume(problem.mesh)
    hF = FacetArea(problem.mesh)
    testt = TestFunction(problem.CR)
    helpp = Function(problem.CR)
    helpp.vector().set_local(np.ones_like(helpp.vector().get_local()))
    a_aux = problem.penalty * (2.*hF('+'))/ (vol('+') + vol('-')) * inner(helpp('+'), testt('+')) * dS + problem.penalty * hF / vol * inner(helpp, testt) * ds
    mat = assemble(a_aux).get_local()

    #creating jump matrix
    mat_jump_1 = dok_matrix((nb_ddl_CR,problem.nb_dof_DEM))
    mat_jump_2 = dok_matrix((nb_ddl_CR,nb_ddl_grad))
    for f in facets(problem.mesh):
        num_global_face = f.index()
        num_global_ddl = dofmap_CR.entity_dofs(problem.mesh, problem.dim - 1, np.array([num_global_face], dtype="uintp"))
        coeff_pen = mat[num_global_ddl][0]
        pos_bary_facet = problem.bary_facets[f.index()] #position barycentre of facet
        
        if len(problem.facet_num.get(f.index())) == 2: #Internal facet
            for c,num_cell,sign in zip(cells(f),problem.facet_num.get(num_global_face),[1., -1.]):
                #filling-out the DG 0 part of the jump
                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,problem.d * num_cell : (num_cell+1) * problem.d] = sign*np.sqrt(coeff_pen)*np.eye(problem.d)
                
                #filling-out the DG 1 part of the jump...
                pos_bary_cell = elt_DG.tabulate_dof_coordinates(c)[0]
                diff = pos_bary_facet - pos_bary_cell
                pen_diff = np.sqrt(coeff_pen)*diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_ddl):
                    for i in range(problem.dim):
                        mat_jump_2[dof_CR,tens_dof_position[(num % problem.d)*problem.d + i]] = sign*pen_diff[i]

        elif len(problem.facet_num.get(f.index())) == 1: #Face sur le bord car n'a qu'un voisin
            #cell part
            #filling-out the DG 0 part of the jump
            num_cell = problem.facet_num.get(f.index())[0]
            mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,problem.d * num_cell : (num_cell+1) * problem.d] = np.sqrt(coeff_pen)*np.eye(problem.d)

            #filling-out the DG 1 part of the jump
            pos_bary_cell = problem.bary_cells.get(num_cell)
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = np.sqrt(coeff_pen)*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                for i in range(problem.dim):
                    mat_jump_2[dof_CR,tens_dof_position[(num % problem.d)*problem.d + i]] = pen_diff[i]

            #boundary facet part
            for v in vertices(f):
                dof_vert = problem.num_ddl_vertex.get(v.index())
                mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,dof_vert[0]:dof_vert[-1]+1] = -np.sqrt(coeff_pen)*np.eye(problem.d) / problem.d
            
    mat_jump = mat_jump_1.tocsr() + mat_jump_2.tocsr() * problem.mat_grad * problem.DEM_to_CR
    return mat_jump.T * mat_jump
