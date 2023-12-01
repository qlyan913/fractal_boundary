# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f: equal to 1 on [1/3,2/3]x[1/3,2/3] and equal to 0 elsewhere
#
# In this example, we would like to evaluate the solution at center of small squares.
#
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
import numpy as np
from firedrake import *
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex3_solver import *
# choose a triangulation
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 4

# stop refinement when sum_eta less than tolerance
tolerance=1e-8

# center points at center of squares of i-th iteration
pp=[[0.5,3/2-(1/3.)**i] for i in range(0,n+1)]
# distance to boundary
x_list=[(1/3.)**i for i in range(0,n+1)]
sum_eta=1
it=0

while sum_eta>tolerance and it<max_iterations:
   it=it+1
   mesh=mesh0
   x, y = SpatialCoordinate(mesh)
   V = FunctionSpace(mesh0, "Lagrange", deg)
   #f=conditional(And(And(And(1./3.<x,x<2./3.),1./3.<y),y<2./3.),1,0)
   f=exp(-20*((x-0.5)**2+(y-0.5)**2))
   g=0.0
   uh = snowsolver(mesh, f,g,V)
   mark, sum_eta,eta_max = Mark(mesh,f,uh,V,tolerance)
   mesh0 = mesh.refine_marked_elements(mark)
   meshplot = triplot(mesh)
   meshplot[0].set_linewidth(0.1)
   meshplot[1].set_linewidth(1)
   plt.xlim(-1, 2)
   plt.axis('equal')
   plt.title('Koch Snowflake Mesh')
   plt.savefig(f"figures/snow_{n}_ref_{it}.pdf")
   plt.close()
   PETSc.Sys.Print(f"refined mesh plot saved to 'figures/snow_{n}_ref_{it}.pdf'.")
   PETSc.Sys.Print("sum_eta is ", sum_eta)

PETSc.Sys.Print(f"refined {it} times")

fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_{n}.png")
PETSc.Sys.Print(f"The plot of force term f is saved to  figures/f_{n}.png")

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




