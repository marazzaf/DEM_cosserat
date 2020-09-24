# coding: utf-8

from dolfin import *
from scipy.sparse import csr_matrix,dok_matrix
import numpy as np
from DEM_cosserat.reconstructions import DEM_to_CR_matrix
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
            raise ValueError('Problem is whether 2d or 3d')
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
        
        #Creating the graph associated with the mesh
        self.Graph = connectivity_graph(self)

        #Computation of gradient matrix for inner penalty term
        self.mat_grad = gradient_matrix(self)

        #DEM reconstructions
        #self.DEM_to_DG_1 = compute_all_reconstruction_matrices(self)
        self.DEM_to_CR = DEM_to_CR_matrix(self)


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

def inner_penalty(problem):
    """Creates the penalty matrix on inner facets to stabilize the DEM."""
    dofmap_tens_DG_0 = problem.W.dofmap()

    #assembling penalty factor
    vol = CellVolume(problem.mesh)
    hF = FacetArea(problem.mesh)
    V = TestFunction(problem.V_CR)
    pen = Constant((problem.penalty_u,problem.penalty_u,problem.penalty_phi))
    U = interpolate(pen, problem.V_CR)
    a_aux = (2.*hF('+'))/ (vol('+') + vol('-')) * inner(U('+'), V('+')) * dS
    mat = assemble(a_aux).get_local()
    mat[mat < 0] = 0 #Putting real zero

    #creating jump matrix
    mat_jump_1 = dok_matrix((problem.nb_dof_CR,problem.nb_dof_DEM))
    mat_jump_2 = dok_matrix((problem.nb_dof_CR,problem.nb_dof_grad))
    for c1,c2 in problem.Graph.edges():
        facet = problem.Graph[c1][c2]
        num_global_face = facet['num']
        num_global_dof_u = facet['dof_CR_u']
        num_global_dof_phi = facet['dof_CR_phi']
        num_global_dof = np.append(num_global_dof_u, num_global_dof_phi)
        coeff_pen = np.sqrt(mat[num_global_dof])
        coeff_pen_u = np.sqrt(mat[num_global_dof][0])
        coeff_pen_phi = np.sqrt(mat[num_global_dof][-1])
        pos_bary_facet = facet['barycentre']
        
        if not facet['bnd']: #Internal facet
            #filling-out the DG 0 part of the jump
            mat_jump_1[num_global_dof,len(num_global_dof) * c1 : (c1+1) * len(num_global_dof)] = np.diag(coeff_pen) # * np.eye(len(num_global_dof))
            mat_jump_1[num_global_dof,len(num_global_dof) * c2 : (c2+1) * len(num_global_dof)] = -np.diag(coeff_pen) #* np.eye(len(num_global_dof))

            for num_cell,sign in zip([c1, c2],[1., -1.]):
                #filling-out the DG 1 part of the jump...
                pos_bary_cell = problem.Graph.nodes[num_cell]['barycentre']
                diff = pos_bary_facet - pos_bary_cell
                pen_diff_u = coeff_pen_u*diff
                pen_diff_phi = coeff_pen_phi*diff
                tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
                for num,dof_CR in enumerate(num_global_dof):
                    for i in range(problem.dim):
                        if num < problem.dim:
                            mat_jump_2[dof_CR,tens_dof_position[num*problem.dim + i]] = sign*pen_diff_u[i]
                        else:
                            mat_jump_2[dof_CR,tens_dof_position[num*problem.dim + i]] = sign*pen_diff_phi[i]

            
    mat_jump = mat_jump_1.tocsr() + mat_jump_2.tocsr() * problem.mat_grad * problem.DEM_to_CR
    return mat_jump.T * mat_jump
