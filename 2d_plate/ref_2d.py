#coding: utf-8

# Computation of the solution in the plate for different meshes
from dolfin import *

def strain(v,psi):
    e = grad(v) + as_tensor(((0, 1), (-1, 0))) * psi
    kappa = grad(psi)
    return e,kappa
    
def computation(mesh, T, E, nu, l):
    #Material parameters
    G = 0.5*E/(1+nu)
    lamda = E*nu / (1+nu) / (1-2*nu)
    M = G*l*l
    Gc = G

    #Computation of stresses
    def stress(e, kappa):
        sigma = lamda * tr(e) * Identity(2) + 2*G * sym(e) + 2*Gc * skew(e)
        mu = 2*self.M * kappa
        return sigma, mu

    #Functionnal spaces
    U = VectorElement("CG", mesh.ufl_cell(), 2) # disp space
    S = FiniteElement("CG", mesh.ufl_cell(), 1) # micro rotation space
    V = FunctionSpace(mesh, MixedElement(U,S))
    print('nb dof CG: %i' % V.dofmap().global_dimension())
    U,S = V.split()
    U_1, U_2 = U.sub(0), U.sub(1)

    # Boundary conditions
    class BotBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[1]) < tol

    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[0]) < tol

    class TopBoundary(SubDomain):
        def inside(self,x,on_boundary):
            tol = 1e-6
            return on_boundary and abs(x[1] - 100) < tol

    t = Constant((0.0, T))
    boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_parts.set_all(0)

    bot_boundary = BotBoundary()
    left_boundary = LeftBoundary()
    top_boundary = TopBoundary()
    top_boundary.mark(boundary_parts, 1)

    ds = Measure('ds')(subdomain_data=boundary_parts)

    u_0 = Constant(0.0)
    left_U_1 = DirichletBC(U.sub(0), u_0, left_boundary)
    bot_U_2 = DirichletBC(U.sub(1), u_0, bot_boundary)
    left_S = DirichletBC(S, u_0, left_boundary)
    bot_S = DirichletBC(S, u_0, bot_boundary)

    bc = [left_U_1, bot_U_2, left_S, bot_S]

    # Variational problem
    u, phi = TrialFunctions(V)
    v, psi = TestFunctions(V)
    trial = stress(u,phi)
    test = strain(v,psi)
    
    a = inner(trial[0], test[0])*dx + inner(trial[1], test[1])*dx
    L = inner(t, v)*ds(1)

    #Solving problem
    U_h = Function(V)
    problem = LinearVariationalProblem(a, L, U_h, bc)
    solver = LinearVariationalSolver(problem)
    solver.solve()
    u_h, psi_h = U_h.split()

    #plot(mesh)
    #plt.show()
    #sys.exit()
    img = plot(u_h[0])
    plt.colorbar(img)
    plt.savefig('FEM/ref_u_x.pdf')
    plt.show()
    img = plot(u_h[1])
    plt.colorbar(img)
    plt.savefig('FEM/ref_u_y.pdf')
    plt.show()
    img = plot(psi_h)
    plt.colorbar(img)
    plt.savefig('FEM/ref_phi.pdf')
    plt.show()

    return U_h


