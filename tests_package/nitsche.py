#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt

# Mesh
nb_elt = 25
mesh = UnitSquareMesh(nb_elt, nb_elt)

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

U = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(U)
v = TestFunction(U)

a = inner(grad(u), grad(v)) * dx

b = 10*v*dx

#Nitsche symmetric test
n = FacetNormal(mesh)
nitsche = - inner(u, dot(grad(v),n)) * ds - inner(v, dot(grad(u),n)) * ds

#Penalty
h = CellDiameter(mesh)
pen = 2/h * inner(u,v) * ds

a += nitsche + pen

sol = Function(U)

bc = DirichletBC(U, Constant(0), boundary)

#solve(a == b, sol, bc)
solve(a == b, sol)

img = plot(sol)
plt.colorbar(img)
plt.show()
