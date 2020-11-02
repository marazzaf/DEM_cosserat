#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt

#Material parameters
nu = 0.3
E = 70e3
lambda_ = nu*E / (1+nu) /(1-2*nu)
mu = 0.5*E/(1+nu)

# Mesh
nb_elt = 80
mesh = UnitSquareMesh(nb_elt, nb_elt)
#mesh = RectangleMesh(Point(-0.5,-0.5), Point(0.5,0.5), nb_elt, nb_elt)

bnd = MeshFunction('size_t', mesh, 1)
bnd.set_all(0)
ds = Measure('ds')(subdomain_data=bnd)

#U = VectorFunctionSpace(mesh, 'CG', 1)
U = VectorFunctionSpace(mesh, 'CR', 1)

u = TrialFunction(U)
v = TestFunction(U)

def sigma(u):
    return lambda_*div(u)*Identity(2) + 2*mu*sym(grad(u))

#Dirichlet BC
u_D = Expression(('a*0.5*(x[0]*x[0]+x[1]*x[1])', 'a*0.5*(x[0]*x[0]+x[1]*x[1])'), a=0.8, degree=2)
bc = DirichletBC(U, u_D, bnd, 0)


a = inner(sigma(u), grad(v)) * dx
volume_load = Expression(('-a*(lmbda+3*mu)', '-a*(lmbda+3*mu)'), a=0.8, lmbda=lambda_, mu=mu, degree=2)
b = inner(volume_load,v) * dx

#Nitsche symmetric test
n = FacetNormal(mesh)
lhs_nitsche = -inner(dot(2*mu*sym(grad(u)), n), v) * ds - inner(lambda_*div(u), dot(v,n)) * ds - inner(dot(2*mu*sym(grad(v)), n), u) * ds - inner(lambda_*div(v), dot(u,n)) * ds #sym
lhs_nitsche_bis = -inner(dot(2*mu*sym(grad(u)), n), v) * ds - inner(lambda_*div(u), dot(v,n)) * ds + inner(dot(2*mu*sym(grad(v)), n), u) * ds + inner(lambda_*div(v), dot(u,n)) * ds #no sym

#Penalty
h = CellDiameter(mesh)
lhs_pen = 2*mu/h * inner(u,v) * ds

a += lhs_nitsche + lhs_pen #sym pen
#a += lhs_nitsche #sym no pen
#a += lhs_nitsche_bis + lhs_pen #no sym

sol = Function(U)

rhs_pen = 2*mu/h * inner(u_D,v) * ds
#rhs_nitsche = inner(dot(2*mu*sym(grad(v)), n), u_D) * ds + inner(lambda_*div(v), dot(u_D,n)) * ds
rhs_nitsche = inner(dot(sigma(v), n), u_D) * ds
#b += rhs_nitsche + rhs_pen #no sym
b += -rhs_nitsche + rhs_pen #sym CR
#b -= rhs_nitsche #sym CG

#solve(a == b, sol, bc)
solve(a == b, sol)

ref = project(u_D, U)

#img = plot(sol[0])
#plt.colorbar(img)
#plt.show()
#img = plot(ref[0])
#plt.colorbar(img)
#plt.show()
#
#img = plot(sol[1])
#plt.colorbar(img)
#plt.show()
#img = plot(ref[1])
#plt.colorbar(img)
#plt.show()

#error = sol - ref
#img = plot(sqrt(error[0]**2 + error[1]**2))
#plt.colorbar(img)
#plt.show()

#errors
err_grad = errornorm(sol, ref, 'H10')
err_L2 = errornorm(sol, ref, 'L2')
print(U.dofmap().global_dimension())
print(err_grad)
print(err_L2)
