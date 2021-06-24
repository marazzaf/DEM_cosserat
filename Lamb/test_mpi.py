#coding: utf-8

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#ref
data = (rank+1)**2
data = comm.gather(data, root=0)
if rank == 0:
    for i in range(size):
        assert data[i] == (i+1)**2
else:
    assert data is None

#test
data = list(np.arange(0,rank+1))
#print(data)
data = comm.gather(data, root=0)
print(data)
aux = []
if rank == 0:
    for l in data:
        print(l)
        aux += l
else:
    assert data is None
    
print(aux)
#del aux
#print(aux)

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
print(data)
if rank == 0:
    print('MTF')
if rank == 0:
    data = {'a': 7, 'b': 3.14}
    req = comm.isend(data, dest=1, tag=11)
    req.wait()
elif rank == 1:
    req = comm.irecv(source=0, tag=11)
    data = req.wait()
print(data)