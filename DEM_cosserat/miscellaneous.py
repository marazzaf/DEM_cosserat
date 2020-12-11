# coding: utf-8
from dolfin import *
from scipy.sparse import csr_matrix
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

def DEM_interpolation(func, problem):
    """Interpolates a function or expression to return a DEM vector containg the interpolation."""

    tot = local_project(func, problem.V_DG).vector().get_local()
    if problem.dim == 2:
        aux = tot.reshape((problem.V_DG.dim() // 3, 3))
        return aux[:,:2].flatten(),aux[:,2],tot
        #return tot
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


def gradient_matrix(problem):
    vol = CellVolume(problem.mesh)

    #variational form gradient
    u_CR,phi_CR = TrialFunctions(problem.V_CR)
    Dv_DG,Dphi_DG = TestFunctions(problem.W)
    a = inner(grad(u_CR), Dv_DG) / vol * dx + inner(grad(phi_CR), Dphi_DG) / vol * dx
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    return csr_matrix((val, col, row))

def lhs_bnd_penalty(problem, subdomain_data, list_Dirichlet_BC=None): #List must contain lists with two parameters: list of components, function (list of components) and possibilty a third: num_domain
    u,phi = TrialFunctions(problem.V_CR) #V_DG1
    v,psi = TestFunctions(problem.V_CR) #V_DG1
    #u,phi = TrialFunctions(problem.V_DG1)
    #v,psi = TestFunctions(problem.V_DG1)
    h = CellDiameter(problem.mesh)
    n = FacetNormal(problem.mesh)

    #stresses
    if problem.dim == 2:
        tr_strains = problem.strains_2d(u,phi)
        tr_sigma,tr_mu = problem.stresses_2d(tr_strains)
        tr_sigma = as_tensor(((tr_sigma[0],tr_sigma[2]), (tr_sigma[3], tr_sigma[1])))
        te_strains = problem.strains_2d(v,psi)
        te_sigma,te_mu = problem.stresses_2d(te_strains)
        te_sigma = as_tensor(((te_sigma[0],te_sigma[2]), (te_sigma[3], te_sigma[1])))
    elif problem.dim == 3:
        tr_strain = problem.strain_3d(u,phi)
        tr_torsion = problem.torsion_3d(phi)
        tr_sigma = problem.stress_3d(tr_strain)
        tr_mu = problem.torque_3d(tr_torsion)
        te_strain = problem.strain_3d(v,psi)
        te_torsion = problem.torsion_3d(psi)
        te_sigma = problem.stress_3d(te_strain)
        te_mu = problem.torque_3d(te_torsion)       

    #Bilinear
    if list_Dirichlet_BC == None: #Homogeneous Dirichlet on all boundary
        bilinear = problem.penalty_u/h * inner(u,v) * ds + problem.penalty_phi/h * inner(phi,psi) * ds - inner(dot(tr_sigma, n), v) * ds - inner(dot(te_sigma, n), u) * ds
    else:
        list_lhs = []
        for BC in list_Dirichlet_BC:
            assert len(BC) == 2 or len(BC) == 3
            if len(BC) == 3:
                domain = BC[2]
                dds = Measure('ds')(subdomain_data=subdomain_data)(domain)
            else:
                dds = Measure('ds')(subdomain_data=subdomain_data)
            component = BC[0]

            if component < problem.dim: #bnd stress
                form_pen = problem.penalty_u / h * u[component] * v[component] * dds - dot(tr_sigma, n)[component] * v[component] * dds - dot(te_sigma, n)[component] * u[component] * dds
            elif component >= problem.dim: #bnd couple stress
                if problem.dim == 3:
                    form_pen = problem.penalty_phi / h * phi[component-problem.dim] * psi[component-problem.dim] * dds - dot(tr_mu, n)[component-problem.dim] * psi[component-problem.dim] * dds - dot(te_mu, n)[component-problem.dim] * phi[component-problem.dim] * dds
                elif problem.dim == 2:
                    form_pen = problem.penalty_phi / h * phi * psi * dds - inner(dot(tr_mu, n), psi) * dds - inner(dot(te_mu, n), phi) * dds
            #Storing new term
            list_lhs.append(form_pen)
                
        #Summing all contributions        
        bilinear = sum(l for l in list_lhs)

    #Assembling matrix
    Mat = assemble(bilinear)
    row,col,val = as_backend_type(Mat).mat().getValuesCSR()
    Mat = csr_matrix((val, col, row), shape=(problem.nb_dof_CR,problem.nb_dof_CR))
    #Mat = csr_matrix((val, col, row), shape=(problem.nb_dof_DG1,problem.nb_dof_DG1))
    
    return problem.DEM_to_CR.T * Mat * problem.DEM_to_CR
    #return problem.DEM_to_DG1.T * Mat * problem.DEM_to_DG1

def rhs_bnd_penalty(problem, subdomain_data, list_Dirichlet_BC): #List must contain lists with three parameters: list of components, function (list of components), num_domain
    #For rhs penalty term computation
    h = CellDiameter(problem.mesh)
    v,psi = TestFunctions(problem.V_CR)
    #v,psi = TestFunctions(problem.V_DG1)
    n = FacetNormal(problem.mesh)

    #stresses
    if problem.dim == 2:
        strains = problem.strains_2d(v,psi)
        sigma,mu = problem.stresses_2d(strains)
        sigma = as_tensor(((sigma[0],sigma[2]), (sigma[3], sigma[1])))
    elif problem.dim == 3:
        strain = problem.strain_3d(v,psi)
        torsion = problem.torsion_3d(psi)
        sigma = problem.stress_3d(strain)
        mu = problem.torque_3d(torsion)
    
    list_L = []
    for BC in list_Dirichlet_BC:
        assert len(BC) == 2 or len(BC) == 3
        if len(BC) == 3:
            domain = BC[2]
            dds = Measure('ds')(subdomain_data=subdomain_data)(domain)
        else:
            dds = Measure('ds')(subdomain_data=subdomain_data)
        imposed_value = BC[1]
        component = BC[0]
        
        #for i,j in enumerate(components):
        if component < problem.dim: #bnd stress
            form_pen = problem.penalty_u / h * imposed_value * v[component] * dds - dot(sigma, n)[component] * imposed_value * dds
        elif component >= problem.dim: #bnd couple stress
            if problem.dim == 3:
                form_pen = problem.penalty_phi / h * imposed_value * psi[component-problem.dim] * dds - dot(mu, n)[component-problem.dim] * imposed_value * dds
            elif problem.dim == 2:
                form_pen = problem.penalty_phi / h * imposed_value * psi * dds - inner(dot(mu, n), imposed_value) * dds
        list_L.append(form_pen)
    L = sum(l for l in list_L)
    L = assemble(L).get_local()
    
    return problem.DEM_to_CR.T * L
    #return problem.DEM_to_DG1.T * L

def energy_error_matrix(problem, subdomain_data):
    ds = Measure('ds')(subdomain_data=subdomain_data)
    h = CellDiameter(problem.mesh)
    h_avg = 0.5 * (h('+') + h('-'))
    u,phi = TrialFunctions(problem.V_DG1)
    v,psi = TestFunctions(problem.V_DG1)
    #form = inner(jump(u), jump(v)) / h_avg * dS + inner(jump(phi), jump(psi)) / h_avg  * dS + inner(u, v) / h * ds + phi * psi / h * ds
    form = inner(u, v) / h * ds + phi * psi / h * ds #Just the boundary...
    Mat = assemble(form)
    row,col,val = as_backend_type(Mat).mat().getValuesCSR()
    Mat = csr_matrix((val, col, row), shape=(problem.nb_dof_DG1,problem.nb_dof_DG1))
    return  problem.DEM_to_DG1.T * Mat * problem.DEM_to_DG1

def lhs_bnd_penalty_test(problem): #Test
    u,phi = TrialFunctions(problem.V_CR)
    v,psi = TestFunctions(problem.V_CR)
    h = CellDiameter(problem.mesh)
    n = FacetNormal(problem.mesh)

    #stresses Nitsche
    tr_strains = problem.strains_2d(u,phi)
    tr_sigma,tr_mu = problem.stresses_2d(tr_strains)
    tr_sigma = as_tensor(((tr_sigma[0],tr_sigma[2]), (tr_sigma[3], tr_sigma[1])))
    te_strains = problem.strains_2d(v,psi)
    te_sigma,te_mu = problem.stresses_2d(te_strains)
    te_sigma = as_tensor(((te_sigma[0],te_sigma[2]), (te_sigma[3], te_sigma[1])))

    #symmetric Nitsche
    nitsche = -inner(dot(tr_sigma, n), v) * ds - inner(dot(te_sigma, n), u) * ds

    #Stresses penalty
    aux = outer(v,n)
    aux = as_vector((aux[0,0], aux[1,1], aux[0,1], aux[1,0]))
    sigma = dot(problem.D, aux)
    sigma = as_tensor(((te_sigma[0],te_sigma[2]), (te_sigma[3], te_sigma[1]))) #2d
    mu = 4*problem.G*problem.l*problem.l * outer(psi, n)

    #Bilinear
    pen_u = 1e2
    pen_phi = pen_u * problem.l * problem.l
    pen = pen_u/h * inner(outer(u,n),sigma) * ds + pen_phi/h * inner(outer(phi,n),mu) * ds
    bilinear = pen + nitsche
    
    
    #Assembling matrix
    Mat = assemble(bilinear)
    row,col,val = as_backend_type(Mat).mat().getValuesCSR()
    Mat = csr_matrix((val, col, row), shape=(problem.nb_dof_CR,problem.nb_dof_CR))
    
    return problem.DEM_to_CR.T * Mat * problem.DEM_to_CR

def rhs_bnd_penalty(problem, imposed_value_u, imposed_value_phi): #Test
    #For rhs penalty term computation
    h = CellDiameter(problem.mesh)
    v,psi = TestFunctions(problem.V_CR)
    n = FacetNormal(problem.mesh)

    #Stresses Nitsche
    strains = problem.strains_2d(v,psi)
    sigma,mu = problem.stresses_2d(strains)
    sigma = as_tensor(((sigma[0],sigma[2]), (sigma[3], sigma[1])))
    
    #symmetric Nitsche
    nitsche = -inner(dot(sigma, n), imposed_value_u) * ds - inner(dot(mu, n), imposed_value_phi) * ds

    #Stresses penalty
    aux = outer(v,n)
    aux = as_vector((aux[0,0], aux[1,1], aux[0,1], aux[1,0]))
    sigma = dot(problem.D, aux)
    sigma = as_tensor(((sigma[0],sigma[2]), (sigma[3], sigma[1]))) #2d
    mu = 4*problem.G*problem.l*problem.l * outer(psi, n)

    #penalty
    pen_u = 1e2
    pen_phi = pen_u * problem.l * problem.l
    pen = pen_u / h * inner(outer(imposed_value_u,n), sigma) * ds + pen_phi / h * inner(outer(imposed_value_phi,n), mu) * ds

    L = assemble(nitsche + pen).get_local()
    
    return problem.DEM_to_CR.T * L
