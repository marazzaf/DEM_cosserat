# coding: utf-8

from dolfin import *
#from scipy.sparse import csr_matrix,diags
import numpy as np
import mpi4py
from petsc4py import PETSc
from DEM_cosserat.reconstructions import *
from DEM_cosserat.mesh_related import *
from DEM_cosserat.miscellaneous import gradient_matrix

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

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
        if rank == 0:
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
        self.WW = self.WW.collapse()
        self.nb_dof_grad = self.W.dofmap().global_dimension()

        #Spaces for penalties
        self.V_DG1 = FunctionSpace(self.mesh, MixedElement(U_DG1,PHI_DG1))
        self.U_DG1,self.PHI_DG1 = self.V_DG1.split()
        self.nb_dof_DG1 = self.V_DG1.dofmap().global_dimension()
        
        #Creating the graph associated with the mesh
        self.Graph = MESH(self)
        #self.Graph = connectivity_graph(self)
        
        #Computation of gradient matrix for inner penalty term
        self.mat_grad = gradient_matrix(self)
        
        ##DEM reconstructions
        #self.DEM_to_CR = DEM_to_CR_matrix(self)
        self.DEM_to_CR = DEM_to_CR_matrix_test(self)
        sys.exit()
        #self.DEM_to_DG1 = DEM_to_DG1_matrix(self)

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

    def micropolar_constants_3d(self, lmbda, G, Gc, L, M, Mc):
        self.lmbda = lmbda
        self.G = G
        self.L = L
        self.Gc = Gc
        self.M = M
        self.Mc = Mc
        
        return      
    
    def strains_2d(self, v, psi):
        e = grad(v) + as_tensor(((0, 1), (-1, 0))) * psi
        kappa = grad(psi)
        return e,kappa

    def stresses_2d(self, strains):
        e,kappa = strains
        eps = as_vector((e[0,0], e[1,1], e[0,1], e[1,0]))
        aux_1 = 2*(1-self.nu)/(1-2*self.nu)
        aux_2 = 2*self.nu/(1-2*self.nu)
        Mat = self.G * as_tensor(((aux_1,aux_2,0,0), (aux_2, aux_1,0,0), (0,0,1+self.a,1-self.a), (0,0,1-self.a,1+self.a)))
        sig = dot(Mat, eps)
        sigma = as_tensor(((sig[0], sig[2]), (sig[3], sig[1])))
        mu = 4*self.G*self.l*self.l * kappa
        return sigma, mu

    def strain_3d(self, v, eta):
        strain = nabla_grad(v)
        strain += as_tensor([ [ 0, -eta[2], eta[1] ] , [ eta[2], 0, -eta[0] ] , [ -eta[1], eta[0], 0 ] ] )
        return strain
    
    def torsion_3d(self, eta):
        return nabla_grad(eta)

    def stress_3d(self, e):
        return self.lmbda * tr(e) * Identity(3) + 2*self.G * sym(e) + 2*self.Gc * skew(e)

    def torque_3d(self, kappa):
        return self.L * tr(kappa) * Identity(3) + 2*self.M * sym(kappa) + 2*self.Mc * skew(kappa)      

    def elastic_bilinear_form(self):
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
            sigma_u = self.stress_3d(epsilon_u)
            mu_u = self.torque_3d(chi_u)

            a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, mu_u)*dx

        A = assemble(a)
        #PETSc mat
        A = as_backend_type(A).mat()
        return self.DEM_to_CR.transpose(PETSc.Mat()) * A * self.DEM_to_CR

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
    ##a_pen = problem.pen / h_avg * inner(jump(u), jump(v)) * dS + problem.pen * problem.l**2/h_avg * inner(jump(phi),jump(psi)) * dS


    #Assembling matrix
    A = assemble(a_pen)
    #PETSc mat
    A = as_backend_type(A).mat()
    return problem.DEM_to_DG1.transpose(PETSc.Mat()) * A * problem.DEM_to_DG1


def mass_matrix(problem, rho=1, I=1): #rho is the volumic mass and I the inertia scalar matrix
    v,psi = TestFunctions(problem.V_DG)
    aux = Constant(['1'] * problem.dim)
    form = rho * inner(aux,v) * dx
    if problem.dim == 3:
        form += rho*I*inner(aux,psi) * dx
    elif problem.dim ==2:
        form += rho*I*psi * dx
    vec = as_backend_type(assemble(form)).vec()

    #creating PETSc mat
    res = PETSc.Mat().create()
    res.setSizes((problem.nb_dof_DEM,problem.nb_dof_DEM))
    res.setUp()
    res.setDiagonal(vec)
    res.assemble() #needed for multiplications

    return res

def mass_matrix_vec(problem, rho=1, I=1): #rho is the volumic mass and I the inertia scalar matrix
    v,psi = TestFunctions(problem.V_DG)
    aux = Constant(['1'] * problem.dim)
    form = rho * inner(aux,v) * dx
    if problem.dim == 3:
        form += rho*I*inner(aux,psi) * dx
    elif problem.dim ==2:
        form += rho*I*psi * dx
    return as_backend_type(assemble(form))
