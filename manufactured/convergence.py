#coding: utf-8

import numpy as np

name = 'convergence_DEM_manufactured.txt'
#name = 'conv_bis.txt'
#name = 'conv_interpolation.txt'
res = np.loadtxt(name)

#manufactured solution
orders_energy = 2 * np.log(res[:-1,3] / res[1:,3]) / np.log(res[1:, 0] / res[:-1,0])
orders_l2 = 2 * np.log(res[:-1,1] / res[1:,1]) / np.log(res[1:, 0] / res[:-1,0])
orders_grad = 2 * np.log(res[:-1,2] / res[1:,2]) / np.log(res[1:, 0] / res[:-1,0])

print(orders_energy)
print(orders_grad)
print(orders_l2)

#test interpolation
#orders_l2_u = 2 * np.log(res[:-1,1] / res[1:,1]) / np.log(res[1:,0] / res[:-1,0])
#orders_l2_phi = 2 * np.log(res[:-1,3] / res[1:,3]) / np.log(res[1:, 2] / res[:-1,2])
#
#print(orders_l2_u)
#print(orders_l2_phi)
