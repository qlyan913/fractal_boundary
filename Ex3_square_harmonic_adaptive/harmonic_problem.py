# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f=exp(-20*((x-0.5)**2+(y-0.5)**2))
#
# In this example, we would like to evaluate the solution at center of small squares.
#
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
import numpy as np
import csv
import statistics
from scipy import stats
from firedrake import *
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex3_solver import *
from firedrake.pyplot import tripcolor

n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
mesh_size=1
N= int(input("Enter the number of segments for estimation on each sides of the bottom  boundary: "))
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
  # meshplot = triplot(mesh)
  # meshplot[0].set_linewidth(0.1)
  # meshplot[1].set_linewidth(1)
  # plt.xlim(-1, 2)
  # plt.axis('equal')
  # plt.title('Koch Snowflake Mesh')
  # plt.savefig(f"figures/snow_{n}_ref_{it}.pdf")
  # plt.close()
  # PETSc.Sys.Print(f"refined mesh plot saved to 'figures/snow_{n}_ref_{it}.pdf'.")
   PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize(), 'sum_eta is ', sum_eta)

PETSc.Sys.Print(f"refined {it} times")

fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_{n}.png")
PETSc.Sys.Print(f"The plot of forcing term f is saved to  figures/f_{n}.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png")
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_{n}.png")

#mesh.name="msh"
#with CheckpointFile(f"solutions/solution_{n}.h5",'w') as afile:
#  afile.save_mesh(mesh)
#  afile.save_function(uh)
#PETSc.Sys.Print(f"The solution is saved to solutions/solution_{n}.h5")



# check the order alpha in u(x)=r^\alpha for x is close to the bottom boundary.
alpha0=[]
c0=[]
# get the vertex on the bottom edge
p3=[np.array([1,0]),1]
p4=[np.array([0,0]),2]
id_pts=2
new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p3,p4]], n)
line_list=line_list+line_list2
x_list=[]
nv_list=[]
# divide the bottom edge into N segments
for L in line_list:
    pts_list,nv=divide_line_N(L,N)
    for pt in pts_list:
       x_list.append([pt,nv])
# estiamte alpha and c
alpha_list=[]
c_list=[]
pt_list=[]
for x in x_list:
    pt=x[0]
    nv=x[1]
    # distance to boundary
    dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in range(5)]
    # sequence of points
    pp=[pt-yy*nv for yy in dy_list]
    uu=uh.at(pp)
    if min(uu)>0:
       dy_list_log=np.log(dy_list)
       uu_log=np.log(uu)
       res = stats.linregress(dy_list_log, uu_log)
       c=exp(res.intercept)
       alpha=res.slope
       alpha_list.append(alpha)
       c_list.append(c)
       pt_list.append(pt)

alpha_mean=statistics.mean(alpha_list)
alpha_std=statistics.stdev(alpha_list)
c_mean=statistics.mean(c_list)
c_std=statistics.stdev(c_list)
PETSc.Sys.Print("Total number of the points estimated on the bottom  boundary is ", len(x_list), ", number of estimated points:", len(alpha_list))
PETSc.Sys.Print("Mean of the alpha is % s, the standard deviation is %s " %(alpha_mean,alpha_std))
PETSc.Sys.Print("Mean of the c is % s, the standard deviation is %s " %(c_mean,c_std))

plt.hist(alpha_list,bins=100)
plt.xlabel('value of alpha')
plt.savefig(f"figures/distribution_alpha_n{n}_N{N}.png")
plt.close()
PETSc.Sys.Print(f"plot of distribution of all estiamted alpha is saved in figures/distribution_alpha_n{n}_N{N}.png.")

plt.hist(c_list,bins=100)
plt.xlabel('value of c')
plt.savefig(f"figures/distribution_c_n{n}_N{N}.png")
plt.close()
PETSc.Sys.Print(f"plot of distribution of all estiamted c is saved in figures/distribution_c_n{n}_N{N}.png.")

with open(f'results/Results_n{n}_N{N}.csv', 'w', newline='') as csvfile:
    fieldnames = ['point','alpha', 'c']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(alpha_list)):
       writer.writerow({'point':pt_list[i],'alpha': alpha_list[i], 'c':c_list[i]})
PETSc.Sys.Print(f"Results of alpha and c are saved to  results/Results_n{n}_N{N}.csv")





