import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import time
from netgen.occ import *
from geogen import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
#n=4
#mesh_size=1
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
ngmsh.Save(f"domain/cantor_{n}.vol")
PETSc.Sys.Print(f"The initial netgen mesh is saved as 'domain/cantor_{n}.vol.")

mesh0 = Mesh(ngmsh)
# Save the mesh
PETSc.Sys.Print(f'Finite element mesh has {mesh0.num_cells()} cells and {mesh0.num_vertices()} vertices.')
# Plot the mesh
plt.figure()
triplot(mesh0)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cantor set Mesh')
plt.show()
plt.savefig(f"figures/cantor_{n}.png")
print(f"plot of the mesh saved as figures/cantor_{n}.png")



