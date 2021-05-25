#coding: utf-8

# Computation of the solution in Cosserat elasticity
from dolfin import *

def computation(mesh, cube, T, nu, mu, Gc, l, num_mesh):
    # Micropolar elastic constants
    lamda = 2*mu*nu / (1-2*nu)
    #torque
    #h3 = 2/5
    M = 2*mu* l*l
    L = 2/3*M
    Mc = M
    #Change these.

    # Strain and torsion
    def strain(v, eta):
        strain = nabla_grad(v)
        strain += as_tensor([ [ 0, -eta[2], eta[1] ] , [ eta[2], 0, -eta[0] ] , [ -eta[1], eta[0], 0 ] ] )
        return strain

    def torsion(eta):
        return nabla_grad(eta)

    # Stress and couple stress
    def stress(e):
        return lamda * tr(e) * Identity(3) + 2*mu * sym(e) + 2*Gc * skew(e)

    def couple(kappa):
        return L * tr(kappa) * Identity(3) + 2*M * sym(kappa) + 2*Mc * skew(kappa)

    # Function Space
    U = VectorElement("CG", mesh.ufl_cell(), 2) # displacement space
    S = VectorElement("CG", mesh.ufl_cell(), 1) # micro rotation space
    V = FunctionSpace(mesh, MixedElement(U,S)) # dim 6
    print('nb dofs FEM: %i' % V.dofmap().global_dimension())
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

    sigma_u = stress(epsilon_u)
    sigma_v = stress(epsilon_v)
    m_u = couple(chi_u)
    m_v = couple(chi_v)

    t = Constant((0.0, T, 0.0))
    a = inner(epsilon_v, sigma_u)*dx + inner(chi_v, m_u)*dx
    L = inner(t, v)*ds(1)

    U_h = Function(V)
    problem = LinearVariationalProblem(a, L, U_h, bcs)
    solver = LinearVariationalSolver(problem)
    solver.parameters['linear_solver'] = 'mumps'
    #solver.parameters['preconditioner'] = 'hypre_amg'
    solver.solve()
    u_h, phi_h = U_h.split()

    #output ref
    file = File('FEM/locking_%i_.pvd' % num_mesh)
    file << u_h
    file << phi_h

    #output stress
    epsilon_u_h = strain(u_h, phi_h)
    sigma_u = stress(epsilon_u_h)
    U = FunctionSpace(mesh, 'CG', 1)
    sigma_yy = project(sigma_u[1,1], U)
    file << sigma_yy

    return u_h,phi_h

T=5e9
E = 3e9
nu = 0.499999
mu = 0.5*E/(1+nu)
cube = 5e-2
mesh_num = 10
mesh = BoxMesh(Point(0., 0., 0.), Point(cube, cube, cube), mesh_num, mesh_num, mesh_num)
computation(mesh, 5e-2, T, nu, mu, 8.84e9, 1e-3, mesh_num)
