#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt

#Material parameters
lambda_ = 1
mu = 2

# Mesh
nb_elt = 25
mesh = UnitSquareMesh(nb_elt, nb_elt)

def boundary(x, on_boundary):
    return on_boundary

U = VectorFunctionSpace(mesh, 'CG', 1)
#U = VectorFunctionSpace(mesh, 'CR', 1)

u = TrialFunction(U)
v = TestFunction(U)

def sigma(u):
    return lambda_*div(u)*Identity(2) + 2*mu*sym(grad(u))

a = inner(sigma(u), grad(v)) * dx

volume_load = Constant((10,5))
b = inner(volume_load,v) * dx

sol = Function(U)

bc = DirichletBC(U, Constant((0,0)), boundary)

solve(a == b, sol, bc)

img = plot(sol[0])
plt.colorbar(img)
plt.show()

img = plot(sol[1])
plt.colorbar(img)
plt.show()
