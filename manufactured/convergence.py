#coding: utf-8

import numpy as np
from scipy.stats import linregress

name = 'convergence_DEM_manufactured.txt'
#name = 'conv_bis.txt'
#name = 'conv_interpolation.txt'
res = np.loadtxt(name)

#manufactured solution
orders_energy = 2 * np.log(res[:-1,3] / res[1:,3]) / np.log(res[1:, 0] / res[:-1,0])
orders_l2 = 2 * np.log(res[:-1,1] / res[1:,1]) / np.log(res[1:, 0] / res[:-1,0])
orders_grad = 2 * np.log(res[:-1,2] / res[1:,2]) / np.log(res[1:, 0] / res[:-1,0])

print(1/orders_l2) #1
print(1/orders_grad) #2
print(np.mean(1/orders_l2))
print(1/orders_energy) #3

#Linear regression
slope, intercept, r_value, p_value, std_err = linregress(np.log(res[:,1]), -np.log(res[:,0]))
print(slope/2)
print(r_value*r_value)

#test interpolation
#orders_l2_u = 2 * np.log(res[:-1,1] / res[1:,1]) / np.log(res[1:,0] / res[:-1,0])
#orders_l2_phi = 2 * np.log(res[:-1,3] / res[1:,3]) / np.log(res[1:, 2] / res[:-1,2])
#
#print(orders_l2_u)
#print(orders_l2_phi)
