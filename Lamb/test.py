#coding: utf-8

from dolfin import *
import sys
sys.path.append('../')
from DEM_cosserat.DEM import *
import mpi4py

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

nb_elt = 5
mesh = UnitSquareMesh(nb_elt, nb_elt)

#Creating the DEM problem
pen = 1
problem = DEMProblem(mesh, pen)

MESH = problem.Graph
data = comm.gather(MESH.list_cells, root=0)
aux = []
if rank == 0:
    print(len(data))
    for l in data:
        aux +=l
print(len(aux))

