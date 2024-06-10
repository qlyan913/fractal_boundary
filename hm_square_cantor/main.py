"""
Solve
   -\Delta u =0 in Omega
with u = 0 on Omega_int, u=1 on Omega_ext

Check the following:
    -\log u_n(x,3^-n)/(n\log 3) --> ? alpha(x,0)
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from firedrake.petsc import PETSc
import numpy as np
import csv
import statistics
from scipy import stats
from firedrake import *
from netgen.geom2d import SplineGeometry
from geogen import *
from hm_solver import *
from firedrake.pyplot import tripcolor
from matplotlib.ticker import PercentFormatter
n=int(input("Enter the number of refinement steps for the cantor set: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
mesh_size=0.2
# choose a triangulation
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# Plot the mesh
plt.figure()
triplot(mesh0)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mesh')
plt.show()
plt.savefig(f"figures/square_{n}.png")
print(f"plot of the mesh saved as figures/square_{n}.png")     

# max of refinement
max_iterations = 40
# stop refinement when sum_eta less than tolerance
tolerance=1*1e-3
uh,f,V=get_solution(mesh0,tolerance,max_iterations,deg)

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png")
plt.close()
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_{n}.png")





