#coding: utf-8

# Plot the error in SCF depending on the elements size
from dolfin import *
import matplotlib.pyplot as plt
from solver_3D import computation
# Parameters
R = 10.0 # radius
cube = 100.0 # dim

T = 1.0 # traction force

nu = 0.49 #0.3 # Poisson's ratio
mu = 1000.0 # shear modulus G
lmbda = ( 2.0 * mu * nu ) / (1.0-2.0*nu) # 1st Lame constant

l = 0.2 # intrinsic length scale
N = 0.93 # coupling parameter

# Analytical solution
def AnalyticalSolution(R, l, nu):
    k = R / l
    eta = 0.2 # ratio of the transverse curvature to the principal curvature
    k_1 = (3.0+eta) / ( 9.0 + 9.0*k + 4.0*k**2 + eta*(3.0 + 3.0*k + k**2) )
    SCF = ( 3.0*(9.0 - 5.0*nu + 6.0*k_1*(1.0-nu)*(1.0+k)) ) / \
          ( 2.0*(7.0 - 5.0*nu + 18.0*k_1*(1.0-nu)*(1.0+k)) )
    return SCF

SCF_a = AnalyticalSolution(R, l, nu)
mesh = Mesh("meshes/3.xml")
hm = mesh.hmax()
SCF_0 = computation(mesh, R, cube, T, nu, mu, lmbda, l, N)
e = abs(SCF_0 - SCF_a) / SCF_a

print('Error: %.5e' % e)
