#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt

# Mesh
nb_elt = 25
mesh = UnitSquareMesh(nb_elt, nb_elt)

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

U = VectorFunctionSpace(mesh, 'CG', 1)
#U = FunctionSpace(mesh, 'CR', 1)

u = TrialFunction(U)
v = TestFunction(U)

def sigma(u):
    lambda_ = 1
    mu = 2
    return lambda_*div(u)*Identity(2) + 2*mu*sym(grad(u))

a = inner(sigma(u), grad(v)) * dx

b = inner(as_vector((10,5)),v) * dx

#Nitsche symmetric test
n = FacetNormal(mesh)
lhs_nitsche = - inner(u, dot(grad(v),n)) * ds - inner(v, dot(grad(u),n)) * ds
#lhs_nitsche_bis = + inner(u, dot(grad(v),n)) * ds - inner(v, dot(grad(u),n)) * ds

#Penalty
h = CellDiameter(mesh)
lhs_pen = 2/h * inner(u,v) * ds

a += lhs_nitsche + lhs_pen #sym pen
#a += lhs_nitsche #sym no pen
#a += lhs_nitsche_bis #nonsym

sol = Function(U)

#bc = DirichletBC(U, Constant(0), boundary)
u_D = Constant((1,0))
rhs_pen = 2/h * inner(u_D,v) * ds
rhs_nitsche = - inner(u_D, dot(grad(v),n)) * ds
#b += rhs_nitsche #no pen
b += rhs_nitsche + rhs_pen

#solve(a == b, sol, bc)
solve(a == b, sol)

img = plot(sol[0])
plt.colorbar(img)
plt.show()

img = plot(sol[1])
plt.colorbar(img)
plt.show()
