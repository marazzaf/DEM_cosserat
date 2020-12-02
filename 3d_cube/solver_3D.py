#coding: utf-8

# Computation of the solution in Cosserat elasticity
from dolfin import *

def computation(mesh, R, cube, T, nu, mu, lmbda, l, N):
    # Micropolar elastic constants
    alpha = ( mu * N**2 ) / (N**2 - 1.0)
    beta = mu * l
    gamma = mu * l**2
    kappa = gamma

    # Strain and torsion
    def strain(v, eta):
        strain = as_tensor([ \
                             [ v[0].dx(0), \
                               v[1].dx(0) - eta[2], \
                               v[2].dx(0) + eta[1] ] , \
                             [ v[0].dx(1) + eta[2], \
                               v[1].dx(1), \
                               v[2].dx(1) - eta[0] ] , \
                             [ v[0].dx(2) - eta[1], \
                               v[1].dx(2) + eta[0], \
                               v[2].dx(2) ] ] )
        return strain

    def torsion(eta):
        torsion = as_tensor([ \
                              [ eta[0].dx(0), eta[1].dx(0), eta[2].dx(0) ], \
                            [ eta[0].dx(1), eta[1].dx(1), eta[2].dx(1) ], \
                            [ eta[0].dx(2), eta[1].dx(2), eta[2].dx(2) ] ])
                            
        return torsion

    # Stress and couple stress
    def stress(lmbda, mu, kappa, epsilon):
        stress = as_tensor([ \
                             [lmbda*epsilon[0,0]+(mu+kappa)*epsilon[0,0]+ mu*epsilon[0,0],
                              \
                              (mu+kappa)*epsilon[0,1] + mu*epsilon[1,0], \
                              (mu+kappa)*epsilon[0,2] + mu*epsilon[2,0] ], \
                            [ (mu+kappa)*epsilon[1,0] + mu*epsilon[0,1], \
                              lmbda*epsilon[1,1] + (mu+kappa)*epsilon[1,1] +
                            mu*epsilon[1,1], \
                              (mu+kappa)*epsilon[1,2] + mu*epsilon[2,1] ], \
                            [ (mu+kappa)*epsilon[2,0] + mu*epsilon[0,2], \
                              (mu+kappa)*epsilon[2,1] + mu*epsilon[1,2], \
                              lmbda*epsilon[2,2] + (mu+kappa)*epsilon[2,2] +
                              mu*epsilon[2,2]] ])
        return stress

    def couple(alpha, beta, gamma, chi):
        couple = as_tensor([ \
                             [ (alpha + beta + gamma)*chi[0,0], \
                               beta*chi[1,0] + gamma*chi[0,1], \
                               beta*chi[2,0] + gamma*chi[0,2] ], \
                             [ beta*chi[0,1] + gamma*chi[1,0], \
                               (alpha + beta + gamma)*chi[1,1], \
                               beta*chi[2,1] + gamma*chi[1,2] ], \
                             [ beta*chi[0,2] + gamma*chi[2,0], \
                               beta*chi[1,2] + gamma*chi[2,1], \
                               (alpha + beta + gamma)*chi[2,2]] ])
        return couple

    # Function Space
    U = VectorElement("CG", mesh.ufl_cell(), 2) # displacement space
    S = VectorElement("CG", mesh.ufl_cell(), 1) # micro rotation space
    V = FunctionSpace(mesh, MixedElement(U,S)) # dim 6
    U, S = V.split()
    U_1, U_2, U_3 = U.split()
    S_1, S_2, S_3 = S.split()

    # Boundary conditions
    class BotBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[1]) < tol
        
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[0]) < tol
            
    class FrontBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[2]) < tol
        
    class TopBoundary(SubDomain):
        def inside(self,x,on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[1] - cube) < tol
        
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_parts.set_all(0)
        
    bot_boundary = BotBoundary()
    left_boundary = LeftBoundary()
    front_boundary = FrontBoundary()
    top_boundary = TopBoundary()
    top_boundary.mark(boundary_parts, 1)

    ds = Measure('ds')(subdomain_data=boundary_parts)

    u_0 = Constant(0.0)

    left_U_1 = DirichletBC(U_1, u_0, left_boundary)
    left_S_2 = DirichletBC(S_2, u_0, left_boundary)
    left_S_3 = DirichletBC(S_3, u_0, left_boundary)

    bot_U_2 = DirichletBC(U_2, u_0, bot_boundary)
    bot_S_1 = DirichletBC(S_1, u_0, bot_boundary)
    bot_S_3 = DirichletBC(S_3, u_0, bot_boundary)

    front_U_3 = DirichletBC(U_3, u_0, front_boundary)
    front_S_1 = DirichletBC(S_1, u_0, front_boundary)
    front_S_2 = DirichletBC(S_2, u_0, front_boundary)

    bcs = [left_U_1, left_S_2, left_S_3, bot_U_2, bot_S_1, \
           bot_S_3, front_U_3, front_S_1, front_S_2]

    # Variational problem
    u, phi = TrialFunctions(V)
    v, eta = TestFunctions(V)
    epsilon_u = strain(u, phi)
    epsilon_v = strain(v, eta)
    chi_u = torsion(phi)
    chi_v = torsion(eta)

    sigma_u = stress(lmbda, mu, kappa, epsilon_u)
    sigma_v = stress(lmbda, mu, kappa, epsilon_v)
    m_u = couple(alpha, beta, gamma, chi_u)
    m_v = couple(alpha, beta, gamma, chi_v)

    t = Constant((0.0, T, 0.0))
    a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx
    L = inner(t, v)*ds(1)

    U_h = Function(V)
    problem = LinearVariationalProblem(a, L, U_h, bcs)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    u_h, phi_h = U_h.split()

    return u_h,phi_h
