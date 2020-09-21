# coding: utf-8
from dolfin import *
import numpy as np
from numpy.linalg import norm
from scipy.sparse import dok_matrix
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
    aux = tot.reshape((problem.V_DG.dim() // 3, 3))

    return aux[:,:2].flatten(),aux[:,2],tot

def assemble_volume_load(load, problem):
    v = TestFunction(problem.DG_0)
    form = inner(load, v) * dx
    L = assemble(form)
    return problem.DEM_to_DG.T * L

def assemble_boundary_load(problem, domain=None, bnd_stress=None, bnd_torque=None):
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
        form = (inner(bnd_torque, eta) + inner(bnd_stress, v)) * ds(domain)
    L = assemble(form)
    return problem.DEM_to_CR.T * L


def nitsche_penalty(problem, list_Dirichlet_BC, D, strain, stress): #List must contain lists with three parameters: list of components, function (list of components), num_domain
    #For rhs penalty term computation
    vol = CellVolume(problem.mesh)
    hF = FacetArea(problem.mesh)
    
    #For consistency term
    n = FacetNormal(problem.mesh)

    #For the rest
    u = TestFunction(problem.V_CR)
    v,eta = TestFunctions(problem.V_CR)
    strains = strain(v,eta)
    stress,couple_stress = stress(D,strains)
    stress = as_tensor([[stress[0],stress[2]],[stress[3],stress[1]]])
    #Que faire en 3d pour le couple stress ?
    
    L = np.zeros(problem.V_CR.dim())
    for BC in list_Dirichlet_BC:
        assert len(BC) == 3    
        domain = BC[2]
        imposed_value = BC[1]
        components = BC[0]
        for i,j in enumerate(components):
            form_pen = problem.penalty * hF / vol * imposed_value[i] * u[j] * ds(domain)
            L += assemble(form_pen).get_local()
            if j < problem.dim: #bnd stress
                form_aux = imposed_value[i] * dot(stress,n)[j]  * ds(domain)
            elif j > problem.dim: #bnd couple stress
                if problem.dim == 3:
                    form_aux = imposed_value[i] * dot(couple_stress,n)[j]  * ds(domain)
                elif problem.dim == 2:
                    form_aux = imposed_value[i] * dot(couple_stress,n)  * ds(domain)
            L += assemble(form_aux).get_local()
            
    return problem.DEM_to_CR.T * L
