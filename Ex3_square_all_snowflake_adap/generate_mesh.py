n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
from geogen import *
print("writing geo")
geo = MakeGeometry(n)
print("generating mesh")
ngmsh = geo.GenerateMesh(maxh=0.4)
print("get firedrake  mesh")
mesh = Mesh(ngmsh)

# Plot the mesh
print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
meshplot = triplot(mesh)
meshplot[0].set_linewidth(0.1)
meshplot[1].set_linewidth(1)
plt.xlim(-1, 2)
plt.axis('equal')
plt.title('Koch Snowflake Mesh')
plt.show()
plt.savefig(f"figures/snow_{n}.pdf")

#with CheckpointFile(f"domain/mesh_{n}.h5",'w') as afile:
 # afile.save_mesh(mesh)
  
