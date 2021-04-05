# coding: utf-8

from dolfin import *
#from scipy.sparse import csr_matrix,diags
import numpy as np
from petsc4py import PETSc
from DEM_cosserat.reconstructions import *
from DEM_cosserat.mesh_related import *
from DEM_cosserat.miscellaneous import gradient_matrix

class DEMProblem:
    """ Class that will contain the basics of a DEM problem from the mesh and the dimension of the problem to reconstrucion matrices and gradient matrix."""
    def __init__(self, mesh, penalty=1., penalty_u=1., penalty_phi=1.):
        self.mesh = mesh
        self.dim = self.mesh.geometric_dimension()
        self.penalty_u = penalty_u
        self.penalty_phi = penalty_phi
        self.pen = penalty

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
        print('nb dof DEM: %i' % self.nb_dof_DEM)

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
    def micropolar_constants(self, E, nu, l, a):
        self.E = E
        self.nu = nu
        self.l = l
        self.a = a

        #computing other parameters
        self.G = 0.5*E/(1+nu)
        return

        
    def micropolar_constants(self, E, nu, l, Gc=0, L=0, Mc=0, incompressible=False):
        self.E = E
        self.nu = nu
        self.l = l
        #used in material law
        self.lamda = E*nu / (1+nu) / (1-2*nu)
        self.G = 0.5*E/(1+nu)
        if Gc > 0:
            self.Gc = Gc
        else:
            self.Gc = self.G
        self.M = self.G*l*l
        #for 3d case
        if self.dim == 3:
            #self.L = ( mu * N*N ) / (N*N - 1)
            if L > 0:
                self.L = L
            else:
                self.L = self.M #best solution?
            if Mc > 0:
                self.Mc = Mc
            else:
                self.Mc = self.M
        if incompressible:
            N = 0.93
            self.lamda = ( 2*self.G*self.nu ) / (1-2*self.nu)
            self.alpha = ( self.G * N**2 ) / (N**2 - 1.0)
            self.beta = self.G * self.l
            self.gamma = self.G * self.l**2
            self.kappa = self.gamma
        return 
        
    
    def strains_2d(self, v, psi):
        e = nabla_grad(v) + as_tensor(((0, -1), (1, 0))) * psi
        kappa = grad(psi)
        return e,kappa

    def stresses_2d(self, strains):
        e,kappa = strains
        if hasattr(self, 'a'):
            aux_1 = 2*(1-self.nu)/(1-2*self.nu)
            aux_2 = 2*self.nu/(1-2*self.nu)
            Mat = G * as_tensor(((aux_1,aux_2,0,0), (0,0,1+self.a,1-self.a), (0,0,1-self.a,1+self.a), (aux_2, aux_1,0,0))) #check if correct
            sigma = dot(Mat, e)
            mu = 4*self.G*self.l*self.l * kappa
        else:
            sigma = self.lamda * tr(e) * Identity(2) + 2*self.G * sym(e) + 2*self.Gc * skew(e)
            mu = 2*self.M * kappa
        return sigma, mu

    def strain_3d(self, v, eta):
        strain = nabla_grad(v)
        strain += as_tensor([ [ 0, -eta[2], eta[1] ] , [ eta[2], 0, -eta[0] ] , [ -eta[1], eta[0], 0 ] ] )
        return strain
    
    def torsion_3d(self, eta):
        return nabla_grad(eta)

    def stress_3d(self, e):
        return self.lamda * tr(e) * Identity(3) + 2*self.G * sym(e) + 2*self.Gc * skew(e)

    def torque_3d(self, kappa):
        return self.L * tr(kappa) * Identity(3) + 2*self.M * sym(kappa) + 2*self.Mc * skew(kappa)      

    def elastic_bilinear_form(self,incompressible=False): #, strain, stress):
        u_CR,psi_CR = TrialFunctions(self.V_CR)
        v_CR,eta_CR = TestFunctions(self.V_CR)

        #Variationnal formulation
        if self.dim == 2:
            def_test = self.strains_2d(v_CR,eta_CR)
            def_trial = self.strains_2d(u_CR, psi_CR)
            stress_trial = self.stresses_2d(def_trial)
            
            a = (inner(def_test[0],stress_trial[0]) + inner(def_test[1],stress_trial[1])) * dx
            
        elif self.dim == 3:
            epsilon_u = self.strain_3d(u_CR, psi_CR)
            epsilon_v = self.strain_3d(v_CR, eta_CR)
            chi_u = self.torsion_3d(psi_CR)
            chi_v = self.torsion_3d(eta_CR)

            if not incompressible:
                sigma_u = self.stress_3d(epsilon_u)
                mu_u = self.torque_3d(chi_u)
            else:
                sigma_u = self.lamda * tr(epsilon_u) * Identity(3) + (self.G+self.kappa) * epsilon_u + self.G * epsilon_u.T
                mu_u = self.alpha * tr(chi_u) * Identity(3) + self.beta * chi_u + self.gamma * chi_u.T

            a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, mu_u)*dx

        A = assemble(a)
        #PETSc mat
        A = as_backend_type(A).mat()
        return self.DEM_to_CR.transpose(PETSc.Mat()) * A * self.DEM_to_CR
        #scipy.sparse mat
        #row,col,val = as_backend_type(A).mat().getValuesCSR()
        #A = csr_matrix((val, col, row))
        #return self.DEM_to_CR.T * A * self.DEM_to_CR

#def inner_penalty_light(problem):
#    """Creates the penalty matrix on inner facets to stabilize the DEM."""
#    h = CellDiameter(problem.mesh)
#    h_avg = 0.5 * (h('+') + h('-'))
#    F = FacetArea(problem.mesh)
#
#    #Average facet jump bilinear form
#    u_DG,phi_DG = TrialFunctions(problem.V_DG1)
#    v_CR,psi_CR = TestFunctions(problem.V_CR)
#    F = FacetArea(problem.mesh)
#    a_jump = sqrt(problem.penalty_u / h_avg / F('+')) * inner(jump(u_DG), v_CR('+')) * dS + sqrt(problem.penalty_phi / h_avg / F('+')) * inner(jump(phi_DG), psi_CR('+')) * dS
#    A = assemble(a_jump)
#    row,col,val = as_backend_type(A).mat().getValuesCSR()
#    A = csr_matrix((val, col, row))
#
#    return problem.DEM_to_DG1.T * A.T * A * problem.DEM_to_DG1

def inner_penalty(problem):
    """Creates the penalty matrix on inner facets to stabilize the DEM."""
    
    #assembling penalty factor
    h = CellDiameter(problem.mesh)
    h_avg = 0.5 * (h('+') + h('-'))
    n = FacetNormal(problem.mesh)

    #Writing penalty bilinear form
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)

    #stresses
    aux = (outer(jump(v),n('+')),outer(jump(psi), n('+')))
    if problem.dim == 2:
        sigma,mu = problem.stresses_2d(aux)
    elif problem.dim == 3:
        sigma = problem.stress_3d(aux[0])
        mu = problem.torque_3d(aux[1])

    #penalty bilinear form
    a_pen = problem.pen / h_avg * inner(outer(jump(u),n('+')), sigma) * dS + problem.pen / h_avg * inner(outer(jump(phi),n('+')), mu) * dS

    #Assembling matrix
    A = assemble(a_pen)
    #PETSc mat
    A = as_backend_type(A).mat()
    return problem.DEM_to_DG1.transpose(PETSc.Mat()) * A.transpose(PETSc.Mat()) * A * problem.DEM_to_DG1

    #scipy.sparse mat
    #row,col,val = as_backend_type(A).mat().getValuesCSR()
    #A = csr_matrix((val, col, row))
    #return problem.DEM_to_DG1.T * A * problem.DEM_to_DG1

def mass_matrix(problem, rho=1, I=1): #rho is the volumic mass and I the inertia scalar matrix
    v,psi = TestFunctions(problem.V_DG)
    aux = Constant(('1', '1', '1'))
    form = rho * (inner(aux,v) + I*inner(aux,psi)) * dx
    vec = as_backend_type(assemble(form)).vec()
    #sys.exit()

    #creating PETSc mat
    res = PETSc.Mat().create()
    res.setSizes((problem.nb_dof_DEM,problem.nb_dof_DEM))
    res.setUp()
    res.setDiagonal(vec)
    res.assemble() #needed for multiplications

    return res

    #np.array
    #return vec.get_local()

    #A = diags(vec.get_local(), 0)
    #A = A.tocsr()
    #petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices,A.data))
    #return PETScMatrix(petsc_mat),min(vec)
