import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import time
from netgen.occ import *
from geogen import *
from Ex2_solver import *
#n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
n=4
mesh_size=1
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
ngmsh.Save(f"domain/cube_{n}.vol")
PETSc.Sys.Print(f"The initial netgen mesh is saved as 'domain/cube_{n}.vol.")

mesh0 = Mesh(ngmsh)
# Save the mesh
PETSc.Sys.Print(f'Finite element mesh has {mesh0.num_cells()} cells and {mesh0.num_vertices()} vertices.')
# Plot the mesh
#fig=plt.figure()
#ax = plt.axes(projection='3d')
#triplot(mesh0,axes=ax)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.title('Koch Snowflake Mesh on Cube')
#plt.savefig(f"domain/cube_{n}.pdf")


File(f"domain/cube_{n}.pvd").write(mesh0)
PETSc.Sys.Print(f"File for visualization in Paraview saved as 'domain/cube_{n}.pvd.")



