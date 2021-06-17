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

print(problem.Graph.list_facets[2].list_cells)
print(problem.Graph.list_facets[2].barycentre)

#for C in test.list_cells:
#    for F in C.list_facets:
#        print(test.list_facets[F].bnd)
