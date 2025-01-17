# coding: utf-8
from dolfin import *
from petsc4py import PETSc
import ufl
import sys

def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def DEM_interpolation(problem, func):
    """Interpolates a function or expression to return a DEM vector containg the interpolation."""

    tot = local_project(func, problem.V_DG).vector()
    if problem.dim == 2:
        aux = tot.get_local().reshape((problem.V_DG.dim() // 3, 3))
        return aux[:,:2].flatten(),aux[:,2],tot.vec()
    elif problem.dim == 3:
        aux = tot.get_local().reshape((problem.V_DG.dim() // 6, 6))
        return aux[:,:3].flatten(),aux[:,3:].flatten(),tot.vec()
    else:
        ValueError('Problem with dimension')

def assemble_volume_load(self, load):
    v = TestFunction(self.V_DG)
    form = inner(load, v) * dx
    return as_backend_type(assemble(form)).vec()

def assemble_boundary_load(problem, domain=None, subdomain_data=None, bnd_stress=None, bnd_torque=None):
    v,eta = TestFunctions(problem.V_CR)
    if bnd_stress is None:
        bnd_stress = Constant(['0'] * problem.dim)
    if bnd_torque is None:
        if problem.dim == 3:
            bnd_torque = Constant(['0'] * problem.dim)
        elif problem.dim == 2:
            bnd_torque = Constant(0.)
    if domain is None:
        form = inner(bnd_stress, v) * ds + inner(bnd_torque, eta) * ds
    else:
        ds = Measure('ds')(subdomain_data=subdomain_data)
        form = (inner(bnd_torque, eta) + inner(bnd_stress, v)) * ds(domain)

    #PETSc
    L = as_backend_type(assemble(form))
    return problem.DEM_to_CR.transpose(PETSc.Mat()) * L.vec()


def gradient_matrix(problem):
    vol = CellVolume(problem.mesh)

    #variational form gradient
    u_CR,phi_CR = TrialFunctions(problem.V_CR)
    Dv_DG,Dphi_DG = TestFunctions(problem.W)
    a = inner(grad(u_CR), Dv_DG) / vol * dx + inner(grad(phi_CR), Dphi_DG) / vol * dx
    A = assemble(a)
    return as_backend_type(A).mat()

def schur_matrices(self):
    aux = list(np.arange(self.nb_dof_cells))
    aux_bis = list(np.arange(self.nb_dof_cells, self.nb_dof_DEM))

    #Get non Dirichlet values
    mat_not_D = dok_matrix((self.nb_dof_cells, self.nb_dof_DEM))
    for (i,j) in zip(range(mat_not_D.shape[0]),aux):
        mat_not_D[i,j] = 1.

    #Get Dirichlet boundary conditions
    mat_D = dok_matrix((self.nb_dof_DEM - self.nb_dof_cells, self.nb_dof_DEM))
    for (i,j) in zip(range(mat_D.shape[0]),aux_bis):
        mat_D[i,j] = 1.
    return mat_not_D.tocsr(), mat_D.tocsr()

def schur_complement(self, A):
    """ Returns the matrices of the Schur complement to remove the values of Dirichlet dofs impose strongly. """
    
    #A_D = self.mat_D * A * self.mat_D.T
    A_not_D = self.mat_not_D * A * self.mat_not_D.T
    B = self.mat_not_D * A * self.mat_D.T
    
    return A_not_D,B

def Dirichlet_CR_dofs(form):
    return set(list(assemble(form).get_local().nonzero()[0]))

#def DEM_aug_interpolation(self, func):
#    """Interpolates a function or expression to return a DEM vector containg the augmented interpolation."""
#    return self.DEM_to_DG.T * local_project(func, self.DG_0).vector().get_local() + self.trace_matrix.T * local_project(func, self.CR).vector().get_local()

def augmented_solution(self, u_DEM, u_D):
    """Adds the information on the boundary to the DEM vector."""
    return self.mat_not_D.T * u_DEM + self.trace_matrix.T * local_project(u_D, self.CR).vector().get_local()


#def lhs_bnd_penalty(problem, subdomain_data, list_Dirichlet_BC=None): #List must contain lists with two parameters: list of components, function (list of components) and possibilty a third: num_domain
#    u,phi = TrialFunctions(problem.V_CR) #V_DG1
#    v,psi = TestFunctions(problem.V_CR) #V_DG1
#    h = CellDiameter(problem.mesh)
#    n = FacetNormal(problem.mesh)
#
#    #stresses
#    if problem.dim == 2:
#        tr_strains = problem.strains_2d(u,phi)
#        tr_sigma,tr_mu = problem.stresses_2d(tr_strains)
#        te_strains = problem.strains_2d(v,psi)
#        te_sigma,te_mu = problem.stresses_2d(te_strains)
#    elif problem.dim == 3:
#        tr_strain = problem.strain_3d(u,phi)
#        tr_torsion = problem.torsion_3d(phi)
#        tr_sigma = problem.stress_3d(tr_strain)
#        tr_mu = problem.torque_3d(tr_torsion)
#        te_strain = problem.strain_3d(v,psi)
#        te_torsion = problem.torsion_3d(psi)
#        te_sigma = problem.stress_3d(te_strain)
#        te_mu = problem.torque_3d(te_torsion)       
#    #Bilinear
#    if list_Dirichlet_BC == None: #Homogeneous Dirichlet on all boundary
#        #bilinear = problem.penalty_u/h * inner(u,v) * ds + problem.penalty_phi/h * inner(phi,psi) * ds - inner(dot(tr_sigma, n), v) * ds - inner(dot(te_sigma, n), u) * ds #ref
#        #test
#        #aux = (outer(v,n),outer(psi, n))
#        #sigma,mu = problem.stresses_2d(aux)
#        #bilinear = problem.pen/h * inner(outer(u,n),sigma) * ds + problem.pen/h/problem.l**2 * inner(outer(phi,n),mu) * ds
#        blinear = -inner(dot(tr_sigma, n), v) * ds + inner(dot(te_sigma, n), u) * ds #consistency and symmetry
#    else:
#        list_lhs = []
#        for BC in list_Dirichlet_BC:
#            assert len(BC) == 2 or len(BC) == 3
#            if len(BC) == 3:
#                domain = BC[2]
#                dds = Measure('ds')(subdomain_data=subdomain_data)(domain)
#            else:
#                dds = Measure('ds')(subdomain_data=subdomain_data)
#            component = BC[0]
#
#            if component < problem.dim: #bnd stress
#                #form_pen = problem.pen*4*problem.G / h * u[component] * v[component] * dds# - dot(tr_sigma, n)[component] * v[component] * dds - dot(te_sigma, n)[component] * u[component] * dds
#                form_pen = -dot(tr_sigma, n)[component] * v[component] * dds + dot(te_sigma, n)[component] * u[component] * dds
#            elif component >= problem.dim: #bnd couple stress
#                if problem.dim == 3:
#                    form_pen = problem.pen * (problem.G+problem.Gc) / h * phi[component-problem.dim] * psi[component-problem.dim] * dds - dot(tr_mu, n)[component-problem.dim] * psi[component-problem.dim] * dds - dot(te_mu, n)[component-problem.dim] * phi[component-problem.dim] * dds
#                elif problem.dim == 2:
#                    #form_pen = problem.pen*4*problem.G / h * phi * psi * dds - inner(dot(tr_mu, n), psi) * dds - inner(dot(te_mu, n), phi) * dds
#                    form_pen = -inner(dot(tr_mu, n), psi) * dds + inner(dot(te_mu, n), phi) * dds
#            #Storing new term
#            list_lhs.append(form_pen)
#                
#        #Summing all contributions        
#        bilinear = sum(l for l in list_lhs)
#
#    #Assembling matrix
#    Mat = assemble(bilinear)
#    Mat = as_backend_type(Mat).mat()
#    return problem.DEM_to_CR.transpose(PETSc.Mat()) * Mat * problem.DEM_to_CR
#
#def rhs_bnd_penalty(problem, subdomain_data, list_Dirichlet_BC): #List must contain lists with three parameters: list of components, function (list of components), num_domain
#    #For rhs penalty term computation
#    h = CellDiameter(problem.mesh)
#    v,psi = TestFunctions(problem.V_CR)
#    n = FacetNormal(problem.mesh)
#
#    #stresses
#    if problem.dim == 2:
#        strains = problem.strains_2d(v,psi)
#        sigma,mu = problem.stresses_2d(strains)
#    elif problem.dim == 3:
#        strain = problem.strain_3d(v,psi)
#        torsion = problem.torsion_3d(psi)
#        sigma = problem.stress_3d(strain)
#        mu = problem.torque_3d(torsion)
#    
#    list_L = []
#    for BC in list_Dirichlet_BC:
#        assert len(BC) == 2 or len(BC) == 3
#        if len(BC) == 3:
#            domain = BC[2]
#            dds = Measure('ds')(subdomain_data=subdomain_data)(domain)
#        else:
#            dds = Measure('ds')(subdomain_data=subdomain_data)
#        imposed_value = BC[1]
#        component = BC[0]
#        
#        #for i,j in enumerate(components):
#        if component < problem.dim: #bnd stress
#            #form_pen = problem.pen*4*problem.G / h * imposed_value * v[component] * dds + dot(sigma, n)[component] * imposed_value * dds
#            form_pen = dot(sigma, n)[component] * imposed_value * dds
#        elif component >= problem.dim: #bnd couple stress
#            if problem.dim == 3:
#                form_pen = problem.pen* (problem.M+problem.Mc) / h * imposed_value * psi[component-problem.dim] * dds - dot(mu, n)[component-problem.dim] * imposed_value * dds
#            elif problem.dim == 2:
#                #form_pen = problem.pen*4*problem.G / h * imposed_value * psi * dds + dot(mu, n)*imposed_value * dds
#                form_pen = dot(mu, n) * imposed_value * dds
#        list_L.append(form_pen)
#    L = sum(l for l in list_L)
#    L = as_backend_type(assemble(L))
#    
#    return problem.DEM_to_CR.transpose(PETSc.Mat()) * L.vec()
#
#def energy_error_matrix(problem, subdomain_data):
#    ds = Measure('ds')(subdomain_data=subdomain_data)
#    h = CellDiameter(problem.mesh)
#    h_avg = 0.5 * (h('+') + h('-'))
#    u,phi = TrialFunctions(problem.V_DG1)
#    v,psi = TestFunctions(problem.V_DG1)
#    #form = inner(jump(u), jump(v)) / h_avg * dS + inner(jump(phi), jump(psi)) / h_avg  * dS + inner(u, v) / h * ds + phi * psi / h * ds
#    form = inner(u, v) / h * ds + phi * psi / h * ds #Just the boundary...
#    Mat = assemble(form)
#    row,col,val = as_backend_type(Mat).mat().getValuesCSR()
#    Mat = csr_matrix((val, col, row), shape=(problem.nb_dof_DG1,problem.nb_dof_DG1))
#    return  problem.DEM_to_DG1.T * Mat * problem.DEM_to_DG1
