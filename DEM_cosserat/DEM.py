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
    h_avg = (vol('+') + vol('-')) / (2.*hF('+'))

    #Writing penalty bilinear form
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)
    a_pen = problem.penalty_u / h_avg * inner(jump(u), jump(v)) * dS + problem.penalty_phi / h_avg * inner(jump(phi), jump(psi)) * dS

    #Assembling matrix
    A = assemble(a_pen)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    return problem.DEM_to_DG1.T * A * problem.DEM_to_DG1


#Add possibility to impose only some components of the vector...
def lhs_nitsche_penalty(problem, list_Dirichlet_BC=None): #List must contain lists with two parameters: list of components, function (list of components) and possibilty a third: num_domain
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)
    vol = CellVolume(problem.mesh)
    hF = FacetArea(problem.mesh)
    n = FacetNormal(problem.mesh)
    h = vol / hF
    strains = strain(u,phi)
    stress,couple_stress = stresses(D,strains)
    if problem.dim == 3:
        stress = as_tensor(((stress[0],stress[1],stress[2]), (stress[3],stress[4],stress[5]), (stress[6],stress[7],stress[8])))
    elif problem.dim == 2:
        stress = as_tensor(((stress[0],stress[1]), (stress[2],stress[3])))

    #Bilinear
    if list_Dirichlet_BC = None:
        bilinear = problem.penalty_u/h * inner(u,v) * ds + problem.penalty_phi/h * inner(phi,psi) * ds + inner(dot(couple_stress,n), psi)*ds + inner(dot(stress,n), v) * ds
        Mat = assemble(bilinear)
        row,col,val = as_backend_type(Mat).mat().getValuesCSR()
        Mat = csr_matrix((val, col, row))
    elif len(list_Dirichlet_BC) >= 2:
        components = BC[0]
        
    return problem.DEM_to_DG1.T * Mat * problem.DEM_to_DG1
    
