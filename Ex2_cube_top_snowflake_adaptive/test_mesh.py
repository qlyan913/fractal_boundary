import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import time
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex2_solver import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))

tolerance = 1e-7
max_iterations = 20
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)

# Plot the mesh
PETSc.Sys.Print(f'Finite element mesh has {mesh0.num_cells()} cells and {mesh0.num_vertices()} vertices.')
outfile = File(f"domain/cube_{n}.pvd")
outfile.write(mesh0)
PETSc.Sys.Print(f"File for visualization in Paraview saved as 'domain/cube_{n}.pvd.")
