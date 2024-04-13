# Qile Yan 2023-02-04
# Solveestimate_all_path.py
#   -\Delta u =0 in Omega
# where Omega= unit square
# u=0 on \partial Omega
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
import numpy as np
from firedrake import *
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex3_solver import *
import statistics 
from scipy import stats
#n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
#deg=int(input("Enter the degree of polynomial in FEM space:"))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
#N= int(input("Enter the number of segments for estimation on the bottom  boundary: "))
N=3**5*32
deg=5
mesh_size=1
# choose a triangulation
geo = MakeGeometry()
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 100
# stop refinement when sum_eta less than tolerance
tolerance=1e-15

uh, mesh, f=get_solution(mesh0,tolerance,max_iterations,deg)

from firedrake.pyplot import tripcolor
fig, axes = plt.subplots()
collection = tripcolor(f, axes=axes)
fig.colorbar(collection);
plt.show()
plt.savefig(f"figures/f.png")
plt.close()
PETSc.Sys.Print(f"The plot of forcing term f is saved to  figures/f.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.xlim(0, 1)
plt.axis('equal')
plt.show()
plt.savefig(f"figures/solution.png")
plt.close()
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution.png")
#outfile = File(f"figures/solution_{n}.pvd")
#outfile.write(uh)

# divide the bottom edge into N segments
x_list= np.linspace(0,1,N)+1/(2*N)
alpha_list=[]
c_list=[]
for i in range(0,len(x_list)-1):
    # distance to boundary
    dy_list=[(1/3.)**5*(1/2)**i for i in range(1,7)]
    # sequence of points
    pp=[[x_list[i],yy] for yy in dy_list]
    uu=uh.at(pp)
    dy_list_log=np.log(dy_list)
    uu_log=np.log(uu)
    res = stats.linregress(dy_list_log, uu_log)
    c=exp(res.intercept)
    alpha=res.slope
    alpha_list.append(alpha)
    c_list.append(c)
    
fig, axes = plt.subplots()
plt.plot(x_list[:-1], alpha_list,marker='o',color='blue')
plt.xlabel('$x$')
plt.ylabel('$alpha$')
plt.savefig(f"figures/distribution_alpha_N{N}.png")
plt.close()
print(f"Plot for alpha saved to figures/distribution_alpha_N{N}.png ")

fig, axes = plt.subplots()
plt.plot(x_list[:-1], c_list,marker='o',color='blue')
plt.xlabel('$x$')
plt.ylabel('$c$')
plt.savefig(f"figures/distribution_c_N{N}.png")
plt.close()
print(f"Plot for alpha saved to figures/distribution_c_N{N}.png ")

