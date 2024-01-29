# Qile Yan 2023-01-29
# Solve
#   -\Delta u =0 in Omega
# where Omega=Q\Q_int, Q=[0,1]x[0,1] with fractal boundary,
# Q_int =[0.4,0.6]x[0.4,0.6]
# u=0 on \partial Q, u=10 on \partial Q_int
# In this example, we would like to evaluate the solution at center of small squares.
#
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
import numpy as np
from firedrake import *
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex3_solver import *
#n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
#deg=int(input("Enter the degree of polynomial in FEM space:"))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
n=8
deg=5
mesh_size=1
# choose a triangulation
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 20

# stop refinement when sum_eta less than tolerance
tolerance=1e-8

# center points at center of squares of i-th iteration
pp=[[0.5,3/2-(1/3.)**i] for i in range(0,n+1)]
# distance to boundary
x_list=[(1/3.)**i for i in range(1,n+1)]

uh=get_solution(mesh0,tolerance,max_iterations,deg)


# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png")
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_{n}.png")

mesh.name="msh"
with CheckpointFile(f"solutions/solution_{n}.h5",'w') as afile:
  afile.save_mesh(mesh)
  afile.save_function(uh)
PETSc.Sys.Print(f"The solution is saved to solutions/solution_{n}.h5")

uu=uh.at(pp)
plt.figure()
plt.plot(x_list,uu,marker='o')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.savefig(f"figures/evaluate_{n}.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution is saved in figures/evaluate_{n}.png.")

tt=x_list
tt=np.array([(x_list[i]/x_list[2])**2.5*uu[2] for i in range(0,len(uu))])
plt.figure()
plt.loglog(x_list,uu,marker='o')
plt.loglog(x_list,tt,marker='v')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.legend(['value of solution','$dist^{2.5}$'])
plt.savefig(f"figures/evaluate_log_{n}.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution in loglog is saved in figures/evaluate_log_{n}.png.")

PETSc.Sys.Print("evaluation of solution  at points:")
PETSc.Sys.Print(pp)
PETSc.Sys.Print("value:")
PETSc.Sys.Print(uu)




