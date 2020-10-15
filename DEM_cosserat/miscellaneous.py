# coding: utf-8
from dolfin import *
from scipy.sparse import csr_matrix
import ufl

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

def DEM_interpolation(func, problem):
    """Interpolates a function or expression to return a DEM vector containg the interpolation."""

    tot = local_project(func, problem.V_DG).vector().get_local()
    if problem.dim == 2:
        aux = tot.reshape((problem.V_DG.dim() // 3, 3))
        return aux[:,:2].flatten(),aux[:,2],tot
    elif problem.dim == 3:
        aux = tot.reshape((problem.V_DG.dim() // 6, 6))
        return aux[:,:3].flatten(),aux[:,3:].flatten(),tot
    else:
        ValueError('Problem with dimension')

def assemble_volume_load(self, load):
    v = TestFunction(self.V_DG)
    form = inner(load, v) * dx
    return assemble(form).get_local()

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
    L = assemble(form)
    return problem.DEM_to_CR.T * L.get_local()


def rhs_nitsche_penalty(problem, list_Dirichlet_BC): #List must contain lists with three parameters: list of components, function (list of components), num_domain
    #For rhs penalty term computation
    h = CellDiameter(problem.mesh)
    
    #For consistency term
    n = FacetNormal(problem.mesh)

    #For the rest
    v,psi = TestFunctions(problem.V_CR) #Use TrialFunctions instead?
    strains = problem.strains(v,psi)
    stress,couple_stress = problem.stresses(strains)
    if problem.dim == 3:
        stress = as_tensor(((stress[0],stress[1],stress[2]), (stress[3],stress[4],stress[5]), (stress[6],stress[7],stress[8])))
    elif problem.dim == 2:
        #stress = as_tensor(((stress[0],stress[1]), (stress[2],stress[3])))
        stress = as_tensor(((stress[0],stress[3]), (stress[2],stress[1])))
    #Que faire en 3d pour le couple stress ?
    
    list_L = []
    for BC in list_Dirichlet_BC:
        assert len(BC) == 2 or len(BC) == 3
        if len(BC) == 3:
            domain = BC[2]
            dds = Measure('ds')(domain)
        else:
            dds = Measure('ds')
        imposed_value = BC[1]
        component = BC[0]
 
        #for i,j in enumerate(components):
        if component < problem.dim: #bnd stress
            form_pen = problem.penalty_u / h * imposed_value * v[component] * dds
            form_pen += imposed_value * dot(stress,n)[component] * dds
        elif component >= problem.dim: #bnd couple stress
            if problem.dim == 3:
                form_pen = problem.penalty_phi / h * imposed_value * psi[component-problem.dim] * dds
                #form_pen += imposed_value * dot(couple_stress,n)[component-problem.dim] * dds
            elif problem.dim == 2:
                form_pen = problem.penalty_phi / h * imposed_value * psi * dds
                form_pen += imposed_value * dot(couple_stress,n) * dds
        list_L.append(form_pen)
    L = sum(l for l in list_L)
    L = assemble(L).get_local()
    
    return problem.DEM_to_CR.T * L

def lhs_nitsche_penalty(problem, list_Dirichlet_BC=None): #List must contain lists with two parameters: list of components, function (list of components) and possibilty a third: num_domain
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)
    n = FacetNormal(problem.mesh)
    h = CellDiameter(problem.mesh)

    #For the rest
    trial_strains = problem.strains(u,phi)
    trial_stress,trial_couple_stress = problem.stresses(trial_strains)
    test_strains = problem.strains(v,psi)
    test_stress,test_couple_stress = problem.stresses(test_strains)
    if problem.dim == 3:
        stress = as_tensor(((stress[0],stress[1],stress[2]), (stress[3],stress[4],stress[5]), (stress[6],stress[7],stress[8])))
    elif problem.dim == 2:
        #trial_stress = as_tensor(((trial_stress[0],trial_stress[1]), (trial_stress[2],trial_stress[3])))
        #test_stress = as_tensor(((test_stress[0],test_stress[1]), (test_stress[2],test_stress[3])))
        trial_stress = as_tensor(((trial_stress[0],trial_stress[3]), (trial_stress[2],trial_stress[1])))
        test_stress = as_tensor(((test_stress[0],test_stress[3]), (test_stress[2],test_stress[1])))

    #Bilinear
    if list_Dirichlet_BC == None: #Homogeneous Dirichlet on all boundary
        bilinear = problem.penalty_u/h * inner(u,v) * ds + problem.penalty_phi/h * inner(phi,psi) * ds - inner(dot(trial_couple_stress,n), psi)*ds - inner(dot(trial_stress,n), v) * ds + inner(dot(test_couple_stress,n), phi)*ds + inner(dot(test_stress,n), u) * ds
    elif len(list_Dirichlet_BC) >= 2:
        list_lhs = []
        for BC in list_Dirichlet_BC:
            assert len(BC) == 2 or len(BC) == 3
            if len(BC) == 3:
                domain = BC[2]
                dds = Measure('ds')(domain)
            else:
                dds = Measure('ds')
            component = BC[0]

            if component < problem.dim: #bnd stress
                form_pen = problem.penalty_u / h * u[component] * v[component] * dds
                form_pen += v[component]  * dot(trial_stress,n)[component] * dds
                form_pen -= u[component]  * dot(test_stress,n)[component] * dds
            elif component >= problem.dim: #bnd couple stress
                if problem.dim == 3:
                    form_pen = problem.penalty_phi / h * phi[component-problem.dim] * psi[component-problem.dim] * dds
                    #form_pen += psi[component-problem.dim] * dot(couple_stress,n)[component-problem.dim] * dds
                elif problem.dim == 2:
                    form_pen = problem.penalty_phi / h * phi * psi * dds
                    form_pen -= psi * dot(trial_couple_stress,n) * dds
                    form_pen += phi * dot(test_couple_stress,n) * dds
            #Storing new term
            list_lhs.append(form_pen)
                
        #Summing all contributions        
        bilinear = sum(l for l in list_lhs)

    #Assembling matrix
    Mat = assemble(bilinear)
    row,col,val = as_backend_type(Mat).mat().getValuesCSR()
    Mat = csr_matrix((val, col, row), shape=(problem.nb_dof_DG1,problem.nb_dof_DG1))
    
    return problem.DEM_to_DG1.T * Mat * problem.DEM_to_DG1

def gradient_matrix(problem):
    vol = CellVolume(problem.mesh)

    #variational form gradient
    u_CR,phi_CR = TrialFunctions(problem.V_CR)
    Dv_DG,Dphi_DG = TestFunctions(problem.W)
    a = inner(grad(u_CR), Dv_DG) / vol * dx + inner(grad(phi_CR), Dphi_DG) / vol * dx
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    return csr_matrix((val, col, row))

def lhs_consistency(problem):
    u, psi = TrialFunctions(problem.V_DG1)
    v, eta = TestFunctions(problem.V_DG1)
    n = FacetNormal(problem.mesh)
    gamma,kappa = problem.strains(u,psi)
    sigma,mu = problem.stresses((gamma,kappa))
    sigma = as_tensor(((sigma[0], sigma[3]), (sigma[2], sigma[1])))
    inner_consistency = -inner(dot(avg(sigma),n('+')), jump(v))*dS - inner(dot(avg(mu),n('+')), jump(eta))*dS
    aux = assemble(inner_consistency)
    row,col,val = as_backend_type(aux).mat().getValuesCSR()
    aux = csr_matrix((val, col, row))
    return problem.DEM_to_DG1.T * aux * problem.DEM_to_DG1
