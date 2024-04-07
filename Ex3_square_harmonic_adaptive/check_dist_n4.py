# Qile Yan 2024-03-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f=exp(-2000*((x-0.5)**2+(y-0.5)**2))
#
# In this example, we would like to evaluate the solution at center of small squares.
# n=4, N=128
# Check the distribution of alpha when x 2/3\leq x \leq 7/9 y \leq -2/9 at bottom
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
n=4
deg=5
N=810
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

l= (1./3.)**n/N
uu_all_list, xl_list,x_list=get_uu(uh,line_list,dy_list,N,l,n)
uu_list,xl_list,alpha_list, c_list,std_list,pt_xlist,pt_ylist=get_alpha(uu_all_list,x_list,xl_list,dy_list)
sub_alpha=[]
for i in range(len(pt_xlist)):
   if pt_xlist[i]>=2./3. and pt_xlist[i]<=7./9. and pt_ylist[i]<=-2./9.:
      sub_alpha.append(alpha_list[i])


plt.hist(sub_alpha,bins=100,range=[0.6,1.25],weights=np.ones(len(sub_alpha))/len(sub_alpha))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.xlabel('value of alpha')
plt.savefig(f"figures/distribution_subalpha_n{n}_N{N}.png")
plt.close()
PETSc.Sys.Print(f"plot of distribution of estiamted alpha is saved in figures/distribution_subalpha_n{n}_N{N}")



