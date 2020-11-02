#coding: utf-8

import numpy as np

name = 'results.txt'
res = np.loadtxt(name)

orders_energy = 2 * np.log(res[1:,1] / res[:-1,1]) / np.log(res[:-1, 0] / res[1:,0])

print(orders_energy)
