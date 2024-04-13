# Qile Yan 2024-02-20
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f=exp(-2000*((x-0.5)**2+(y-0.5)**2)) point source as in  https://github.com/firedrakeproject/firedrake/discussions/2510
#
# In this example, we would like to evaluate the solution at center of small squares.
#
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
from Ex3_solver import *
from firedrake.pyplot import tripcolor
from matplotlib.ticker import PercentFormatter
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
mesh_size=1
#N= int(input("Enter the number of segments for estimation on each sides of the bottom  boundary: "))
N=3**(7-n)*32
# choose a triangulation
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 100
# stop refinement when sum_eta less than tolerance
tolerance=5*1e-17

uh,f,V=harmonic_get_solution(mesh0,tolerance,max_iterations,deg)

fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_{n}.png")
plt.close()
PETSc.Sys.Print(f"The plot of forcing term f is saved to  figures/f_{n}.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png")
plt.close()
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_{n}.png")

#mesh.name="msh"
#with CheckpointFile(f"solutions/solution_{n}.h5",'w') as afile:
#  afile.save_mesh(mesh)
#  afile.save_function(uh)
#PETSc.Sys.Print(f"The solution is saved to solutions/solution_{n}.h5")

PETSc.Sys.Print("Calculating the alpha for points on the bottom boundary ...")
# check the order alpha in u(x)=r^\alpha for x is close to the bottom boundary.
alpha0=[]
c0=[]
# get the vertex on the bottom edge
p3=[np.array([1,0]),1]
p4=[np.array([0,0]),2]
id_pts=2
new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p3,p4]], n)
line_list=line_list+line_list2
d_ins = np.linspace(0,6,20)
dy_list=[0.5*(1/3.)**7*(1/2.)**i for i in d_ins]
l= (1./3.)**n/N

#alpha_list,c_list,xl_list,uu_all_list,std_list,pt_xlist,pt_ylist=get_alpha(uh,line_list,dy_list,N,l)
uu_all_list, xl_list,x_list=get_uu(uh,line_list,dy_list,N,l,n)
uu_list,xl_list,alpha_list, c_list,std_list,pt_xlist,pt_ylist=get_alpha(uu_all_list,x_list,xl_list,dy_list)


alpha_mean=statistics.mean(alpha_list)
alpha_std=statistics.stdev(alpha_list)
c_mean=statistics.mean(c_list)
c_std=statistics.stdev(c_list)
PETSc.Sys.Print("Mean of the alpha is % s, the standard deviation is %s " %(alpha_mean,alpha_std))
PETSc.Sys.Print("Mean of the c is % s, the standard deviation is %s " %(c_mean,c_std))

plt.hist(alpha_list,bins=100,range=[0.95,1.05],weights=np.ones(len(alpha_list))/len(alpha_list))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('value of alpha')
plt.savefig(f"figures/distribution_alpha_n{n}_N{N}.png")
plt.close()
PETSc.Sys.Print(f"plot of distribution of all estiamted alpha is saved in figures/distribution_alpha_n{n}_N{N}.png.")


fig=plt.figure()
indx_list=sort_index(line_list,pt_xlist,pt_ylist) # this function is in geogen.py
im=plot_colourline(pt_xlist,pt_ylist,alpha_list,indx_list)
#im.set_clim(0.6,1.25)
fig.colorbar(im)
plt.xlim(0,1)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f"figures/alpha_n{n}_N{N}.png",dpi=500)
plt.close()
PETSc.Sys.Print(f"plot of alpha with color is saved in figures/alpha_n{n}_N{N}.png")

fig=plt.figure()
indx_list=sort_index(line_list,pt_xlist,pt_ylist) # this function is in geogen.py
im=plot_colourline_std(pt_xlist,pt_ylist,std_list,indx_list)
fig.colorbar(im)
plt.xlim(0,1)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(f"figures/alpha_std_n{n}_N{N}.png",dpi=500)
plt.close()
PETSc.Sys.Print(f"plot of std of alpha with color is saved in figures/alpha_std_cm_n{n}_N{N}.png")

#plt.figure()
#plt.hist(c_list,bins=100)
#plt.xlabel('value of c')
#plt.savefig(f"figures/distribution_c_n{n}_N{N}.png")
#plt.close()
#PETSc.Sys.Print(f"plot of distribution of all estiamted c is saved in figures/distribution_c_n{n}_N{N}.png.")

#with open(f'results/Results_n{n}_N{N}.csv', 'w', newline='') as csvfile:
#    fieldnames = ['x','y','alpha','std_alpha', 'c']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    writer.writeheader()
#    for i in range(len(alpha_list)):
#       writer.writerow({'x':pt_xlist[i],'y':pt_ylist[i],'alpha': alpha_list[i],'std_alpha':std_list[i], 'c':c_list[i]})
#PETSc.Sys.Print(f"Results of alpha and c are saved to  results/Results_n{n}_N{N}.csv")





