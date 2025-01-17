# coding: utf-8

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

L = 6e-2
thick = Constant(L/10)
width = Constant(L/10)
E = Constant(3e9)
nu = Constant(0.3)

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

class RightEnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > L/2
def left_end(x, on_boundary):
    return on_boundary and x[0] < L/2

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
right_boundary = RightEnd()
right_boundary.mark(boundary_parts, 1)
ds = Measure('ds')(subdomain_data=boundary_parts)

k_form = EI*inner(grad(theta_), grad(dtheta))*dx + kappa*GS*dot(grad(w_)[0]-theta_, grad(dw)[0]-dtheta)*dx
l_form = Constant(1e9)*theta_*ds(1)

bc = [DirichletBC(V.sub(0), Constant(0.), left_end), DirichletBC(V.sub(1), Constant(0.), left_end)]

u = Function(V)
solve(k_form == l_form, u, bc)
w,theta = u.split()

plot(w)
plt.show()
plot(theta)
plt.show()



