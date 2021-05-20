# coding: utf-8

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

L = 1.
thick = Constant(0.03)
width = Constant(0.01)
E = Constant(70e3)
nu = Constant(0.)

EI = E*width*thick**3/12
GS = E/2/(1+nu)*thick*width
kappa = Constant(5./6.)


N = 100
mesh = IntervalMesh(N, 0, L) 

U = FiniteElement("CG", mesh.ufl_cell(), 2)
T = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, U*T)

u_ = TestFunction(V)
du = TrialFunction(V)
(w_, theta_) = split(u_)
(dw, dtheta) = split(du)


k_form = EI*inner(grad(theta_), grad(dtheta))*dx + kappa*GS*dot(grad(w_)[0]-theta_, grad(dw)[0]-dtheta)*dx
l_form = Constant(1.e-5)*u_[0]*dx #add Neumann stress on rhs

def right_end(x, on_boundary):
    return on_boundary and near(x[0], L)
def left_end(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc = [DirichletBC(V.sub(0), Constant(0.), left_end), DirichletBC(V.sub(1), Constant(0.), left_end), DirichletBC(V.sub(0), Constant(1.), right_end)] #change bc

u = Function(V)
solve(k_form == l_form, u, bc)
w,theta = u.split()

plot(w)
plt.show()



