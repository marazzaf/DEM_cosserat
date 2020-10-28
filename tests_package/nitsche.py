#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt

# Mesh
nb_elt = 25
mesh = UnitSquareMesh(nb_elt, nb_elt)

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

#U = FunctionSpace(mesh, 'CG', 1)
U = FunctionSpace(mesh, 'CR', 1)

u = TrialFunction(U)
v = TestFunction(U)

a = inner(grad(u), grad(v)) * dx

b = 10*v*dx

#Nitsche symmetric test
n = FacetNormal(mesh)
lhs_nitsche = - inner(u, dot(grad(v),n)) * ds - inner(v, dot(grad(u),n)) * ds
#lhs_nitsche_bis = + inner(u, dot(grad(v),n)) * ds - inner(v, dot(grad(u),n)) * ds

#Penalty
h = CellDiameter(mesh)
lhs_pen = 2/h * inner(u,v) * ds

#a += lhs_nitsche + lhs_pen #sym pen
a += lhs_nitsche #sym no pen
#a += lhs_nitsche_bis #nonsym

sol = Function(U)

#bc = DirichletBC(U, Constant(0), boundary)
u_D = Constant(1)
rhs_pen = 2/h * inner(u_D,v) * ds
rhs_nitsche = - inner(u_D, dot(grad(v),n)) * ds
b += rhs_nitsche

#solve(a == b, sol, bc)
solve(a == b, sol)

img = plot(sol)
plt.colorbar(img)
plt.show()
