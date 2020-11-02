#coding: utf-8

import numpy as np

#name = 'res_ref.txt'
#name = 'res_nitsche_sym_pen.txt'
#name = 'res_nitsche_sym.txt'
name = 'res_no_nitsche.txt'
res = np.loadtxt(name)

orders_energy = 2 * np.log(res[1:,1] / res[:-1,1]) / np.log(res[:-1, 0] / res[1:,0])
orders_l2 = 2 * np.log(res[1:,2] / res[:-1,2]) / np.log(res[:-1, 0] / res[1:,0])

print(orders_energy)
print(orders_l2)
