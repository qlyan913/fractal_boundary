# Qile Yan 2024-03-25
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f=exp(-2000*((x-0.5)**2+(y-0.5)**2))
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
#n=8
#deg=5

mesh_size=1
# choose a triangulation
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 100
# stop refinement when sum_eta less than tolerance
tolerance=1e-10
uh,f,V=harmonic_get_solution(mesh0,tolerance,max_iterations,deg)

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
# distance to boundary
d_ins = np.linspace(0,7,20)
dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in d_ins]

N_all=[2*2**i for i in range(7)]
l_list=[] # size of segments
ms_list=[]
ms_sum_list=[]
ms_u_list=[]
ms_u_sum_list=[]
for N in N_all:
   l= (1./3.)**n/N
   l_list.append(l)
   ms=0
   ms_sum=0
   ms_u=0
   ms_u_sum=0
   alpha_list,c_list,xl_list,uu_all_list=get_alpha(uh,line_list,dy_list,N,l)[0:4]
   for i in range(len(alpha_list)):
#     plot_regression(f"reg_figs/n{n}/reg_n{n}_N{N}_i{i}.png",uu_all_list[i],c_list[i],dy_list,alpha_list[i],uh.at(xl_list[i]),l/2.)
     ms_sum=ms_sum+c_list[i]*(l/2.)**alpha_list[i]
     ms_u_sum=ms_u_sum+uh.at(xl_list[i]) 
     if alpha_list[i]>1:
        ms=ms+c_list[i]*(l/2.)**alpha_list[i]
        ms_u=ms_u+uh.at(xl_list[i])
#        plot_regression(f"reg_figs/n{n}/reg_n{n}_N{N}_i{i}.png",uu_all_list[i],c_list[i],dy_list,alpha_list[i],uh.at(xl_list[i]),l/2.)
   ms_list.append(ms)
   ms_sum_list.append(ms_sum)
   ms_u_list.append(ms_u)
   ms_u_sum_list.append(ms_u_sum)
   i_tmp=1
  # plot_regression(f"reg_figs/n{n}/reg_n{n}_N{N}.png",uu_all_list[i_tmp],c_list[i_tmp],dy_list,alpha_list[i_tmp],uh.at(xl_list[i_tmp]),l/2.)

plt.figure()
plt.loglog(l_list,ms_list,'b.')
plt.loglog(l_list,ms_u_list,'k*')
plt.xlabel('size of segments, $|l|$')
plt.legend([r'$\sum_{\alpha_l>1}c_l|l/2|^{\alpha_l}$',r'$\sum_{\alpha_l>1}u(x_l)$'])
plt.savefig(f"figures/hm_n{n}.png")
plt.close()
print(f"plot of harmonic measure of edges  with alpha larger than 1 is saved in figures/hm_n{n}.png")

ms_p=[]
ms_u_p=[]
for i in range(len(ms_list)):
   ms_p.append(ms_list[i]/ms_sum_list[i])
   ms_u_p.append(ms_u_list[i]/ms_u_sum_list[i])
plt.figure()
plt.loglog(l_list,ms_p,'b.')
plt.loglog(l_list,ms_u_p,'k*')
plt.xlabel('size of segments, $|l|$')
plt.legend([r'$\sum_{\alpha_l>1}c_l|l/2|^{\alpha_l}/\sum_{l}c_l|l/2|^{\alpha_l}$',r'$\sum_{\alpha_l>1}u(x_l)/\sum_{l}u(x_l)$'])
plt.savefig(f"figures/hm_n{n}_p.png")
plt.close()
print(f"plot of percentage of  harmonic measure of edges  with alpha larger than 1 is saved in figures/hm_n{n}_p.png")


