# coding: utf-8
from dolfin import *
import numpy as np
from numpy.linalg import norm
from scipy.sparse import dok_matrix

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

    aux = local_project(func, problem.V_DG).vector().get_local()
    aux = aux.reshape((problem.V_DG.dim() // 3, 3))

    return aux[:,:2].flatten(),aux[:,2]

def Dirichlet_BC(form, DEM_to_CG):
    L = assemble(form)
    return DEM_to_CG.T * L.get_local()

def assemble_volume_load(load, problem):
    v = TestFunction(problem.DG_0)
    form = inner(load, v) * dx
    L = assemble(form)
    return problem.DEM_to_DG.T * L


def schur_matrices(A_BC):
    nb_ddl_ccG = A_BC.shape[0]
    l = A_BC.nonzero()[0]
    aux = set(l) #contains number of Dirichlet dof
    nb_ddl_Dirichlet = len(aux)
    aux_bis = set(range(nb_ddl_ccG))
    aux_bis = aux_bis.difference(aux) #contains number of vertex non Dirichlet dof
    sorted(aux_bis) #sort the set

    #Get non Dirichlet values
    mat_not_D = dok_matrix((nb_ddl_ccG - nb_ddl_Dirichlet, nb_ddl_ccG))
    for (i,j) in zip(range(mat_not_D.shape[0]),aux_bis):
        mat_not_D[i,j] = 1.

    #Get Dirichlet boundary conditions
    mat_D = dok_matrix((nb_ddl_Dirichlet, nb_ddl_ccG))
    for (i,j) in zip(range(mat_D.shape[0]),aux):
        mat_D[i,j] = 1.
    return mat_not_D.tocsr(), mat_D.tocsr()

def schur_complement(L, u_BC, B, problem):
    L_not_D = problem.mat_not_D * L
    u_BC_interpolate = interpolate(u_BC, problem.CG).vector().get_local()
    u_BC_interpolate = problem.mat_D * problem.DEM_to_CG.T * u_BC_interpolate
    L_not_D = L_not_D - B * u_BC_interpolate
    return L_not_D,u_BC_interpolate

def complete_solution(u_reduced, u_BC, problem):
    return problem.mat_not_D.T * u_reduced + problem.mat_D.T * u_BC
