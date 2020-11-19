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
        return G * as_matrix([[a,b,0,0], [b,a,0,0], [0,0,c,d], [0,0,d,c]])

    def micropolar_constants(nu, mu, lmbda, l, N):
        self.mu = mu
        self.nu = nu
        self.lmbda = lmbda
        self.l = l
        self.N = N
        self.alpha = ( mu * N*N ) / (N*N - 1)
        self.beta = mu * l
        self.gamma = mu * l*l
        self.kappa = self.gamma
        return 
        
    def strains_2d(self, v, eta):
        gamma = as_vector([v[0].dx(0), v[1].dx(1), v[1].dx(0) - eta, v[0].dx(1) + eta])
        kappa = grad(eta)
        return gamma, kappa

    def stresses_2d(self, strains):
        gamma,kappa = strains
        sigma = dot(self.D, gamma)
        mu = 4*self.G*self.l*self.l * kappa
        return sigma, mu

    def strain_3d(v, eta):
        strain = nabla_grad(v)
        strain += as_tensor([ [ 0, -eta[2], eta[1] ] , \
[ eta[2], 0, -eta[0] ] , [ -eta[1], eta[0], 0 ] ] )
        return strain

    def torsion_3d(eta):
        return nabla_grad(eta)

    def stress_3d(epsilon):
        stress = as_tensor([ \
                             [self.lmbda*epsilon[0,0]+(self.mu+self.kappa)*epsilon[0,0]+ self.mu*epsilon[0,0],
                              \
                              (self.mu+self.kappa)*epsilon[0,1] + self.mu*epsilon[1,0], \
                              (self.mu+self.kappa)*epsilon[0,2] + self.mu*epsilon[2,0] ], \
                            [ (self.mu+self.kappa)*epsilon[1,0] + self.mu*epsilon[0,1], \
                              lmbda*epsilon[1,1] + (self.mu+self.kappa)*epsilon[1,1] +
                            self.mu*epsilon[1,1], \
                              (self.mu+self.kappa)*epsilon[1,2] + self.mu*epsilon[2,1] ], \
                            [ (self.mu+self.kappa)*epsilon[2,0] + self.mu*epsilon[0,2], \
                              (self.mu+self.kappa)*epsilon[2,1] + self.mu*epsilon[1,2], \
                              lmbda*epsilon[2,2] + (self.mu+self.kappa)*epsilon[2,2] +
                              self.mu*epsilon[2,2]] ])
        return stress

    def torque_3d(chi):
        torque = as_tensor([ \
                             [ (self.alpha + self.beta + self.gamma)*chi[0,0], \
                               self.beta*chi[1,0] + self.gamma*chi[0,1], \
                               self.beta*chi[2,0] + self.gamma*chi[0,2] ], \
                             [ self.beta*chi[0,1] + self.gamma*chi[1,0], \
                               (self.alpha + self.beta + self.gamma)*chi[1,1], \
                               self.beta*chi[2,1] + self.gamma*chi[1,2] ], \
                             [ self.beta*chi[0,2] + self.gamma*chi[2,0], \
                               self.beta*chi[1,2] + self.gamma*chi[2,1], \
                               (self.alpha + self.beta + self.gamma)*chi[2,2]] ])
        return torque

    def elastic_bilinear_form(self): #, strain, stress):
        u_CR,psi_CR = TrialFunctions(self.V_CR)
        v_CR,eta_CR = TestFunctions(self.V_CR)

        #Variationnal formulation
        if self.d == 2:
            def_test = self.strains_2d(v_CR,eta_CR)
            def_trial = self.strains_2d(u_CR, psi_CR)
            stress_trial = self.stresses_2d(def_trial)
            a = (inner(def_test[0],stress_trial[0]) + inner(def_test[1],stress_trial[1])) * dx
        elif self.d == 3:
            epsilon_u = strain_3d(u_CR, psi_CR)
            epsilon_v = strain_3d(v_CR, eta_CR)
            chi_u = torsion_3d(psi_CR)
            chi_v = torsion_3d(eta_CR)

            sigma_u = stress_3d(epsilon_u)
            sigma_v = tress_3d(epsilon_v)
            m_u = torque_3d(chi_u)
            m_v = torque_3d(chi_v)

            a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx

        A = assemble(a)
        row,col,val = as_backend_type(A).mat().getValuesCSR()
        A = csr_matrix((val, col, row))
        return self.DEM_to_CR.T * A * self.DEM_to_CR

def inner_consistency(problem):
    """Creates the penalty matrix on inner facets to stabilize the DEM."""
    
    #assembling penalty factor
    h = CellDiameter(problem.mesh)
    h_avg = 0.5 * (h('+') + h('-'))
    n = FacetNormal(problem.mesh)

    #Writing penalty bilinear form
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)
    #u,phi = TrialFunctions(problem.V_CR)
    #v,psi = TestFunctions(problem.V_CR)

    #stresses
    tr_strains = problem.strains(u,phi)
    tr_sigma,tr_mu = problem.stresses(tr_strains)
    tr_sigma = as_tensor(((tr_sigma[0],tr_sigma[2]), (tr_sigma[3], tr_sigma[1]))) #2d
    te_strains = problem.strains(v,psi)
    te_sigma,te_mu = problem.stresses(te_strains)
    te_sigma = as_tensor(((te_sigma[0],te_sigma[2]), (te_sigma[3], te_sigma[1]))) #2d
    
    #a_pen = problem.penalty_u / h_avg * inner(jump(u), jump(v)) * dS + problem.penalty_phi / h_avg * inner(jump(phi), jump(psi)) * dS
    a_pen = - inner(dot(avg(tr_sigma), n('+')), jump(v)) * dS - inner(dot(avg(tr_mu), n('+')), jump(psi)) * dS - inner(dot(avg(te_mu), n('+')), jump(phi)) * dS - inner(dot(avg(te_sigma), n('+')), jump(u)) * dS

    #Assembling matrix
    A = assemble(a_pen)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    return problem.DEM_to_DG1.T * A * problem.DEM_to_DG1
    #return problem.DEM_to_CR.T * A * problem.DEM_to_CR

def inner_penalty_light(problem):
    """Creates the penalty matrix on inner facets to stabilize the DEM."""
    h = CellDiameter(problem.mesh)
    h_avg = 0.5 * (h('+') + h('-'))
    F = FacetArea(problem.mesh)

    #Average facet jump bilinear form
    u_DG,phi_DG = TrialFunctions(problem.V_DG1)
    v_CR,psi_CR = TestFunctions(problem.V_CR)
    F = FacetArea(problem.mesh)
    a_jump = sqrt(problem.penalty_u / h_avg / F('+')) * inner(jump(u_DG), v_CR('+')) * dS + sqrt(problem.penalty_phi / h_avg / F('+')) * inner(jump(phi_DG), psi_CR('+')) * dS
    A = assemble(a_jump)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    return problem.DEM_to_DG1.T * A.T * A * problem.DEM_to_DG1
