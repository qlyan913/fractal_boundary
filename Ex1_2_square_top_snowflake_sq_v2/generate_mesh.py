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
ngmsh.Save(f"domain/square_{n}.vol")
PETSc.Sys.Print(f"The initial netgen mesh is saved as 'domain/cube_{n}.vol.")

mesh0 = Mesh(ngmsh)
# Save the mesh
PETSc.Sys.Print(f'Finite element mesh has {mesh0.num_cells()} cells and {mesh0.num_vertices()} vertices.')
# Plot the mesh
plt.figure()
meshplot=triplot(mesh0)
meshplot[0].set_linewidth(0.1)
meshplot[1].set_linewidth(1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Koch Snowflake Mesh')
plt.show()
plt.savefig(f"figures/snow_{n}.pdf")
print(f"plot of the mesh saved as figures/snow_{n}.pdf")



