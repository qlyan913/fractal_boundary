# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f=exp(-2000*((x-0.5)**2+(y-0.5)**2))
#
# In this script, we calculate the harmonic measure of the bottom
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
#n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
#deg=int(input("Enter the degree of polynomial in FEM space:"))
deg=5
def get_hm_subset(n,N):
   # distance to boundary
   d_ins = np.linspace(0,6,10)
   dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in d_ins]
   mesh_size=1
   # choose a triangulation
   geo = MakeGeometry(n)
   ngmsh = geo.GenerateMesh(maxh=mesh_size)
   mesh0 = Mesh(ngmsh)
   # max of refinement  
   max_iterations = 100
   # stop refinement when sum_eta less than tolerance
   tolerance=1e-17
   uh,f,V=harmonic_get_solution(mesh0,tolerance,max_iterations,deg)
   PETSc.Sys.Print("Calculating the alpha for points on the bottom boundary ...")
   p3=[np.array([1,0]),1]
   p4=[np.array([0,0]),2]
   id_pts=2
   new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p3,p4]], n)
   line_list=line_list+line_list2
   l= (1./3.)**n/N
   ms_sum=0
   ms_u_sum=0
   ns=0
   uu_all_list, xl_list,x_list=get_uu(uh,line_list,dy_list,N,l,n)
   uu_list,xl_list,alpha_list, c_list,std_list,pt_xlist,pt_ylist=get_alpha(uu_all_list,x_list,xl_list,dy_list)
   sub_ptx=[]
   sub_pty=[]
   for i in range(len(alpha_list)):
      ns=ns+1
      ms_sum=ms_sum+c_list[i]*(l/2.)**alpha_list[i]
      ms_u_sum=ms_u_sum+uh.at(xl_list[i])
   return ms_sum, ms_u_sum, ns, sub_ptx,sub_pty

ms=[]
ms_u=[]
for n in range(1,7):
   N=10*3**(8-n)
   ms_sum, ms_u_sum, ns,sub_ptx,sub_pty = get_hm_subset(n,N)
   ms.append(ms_sum)
   ms_u.append(ms_u_sum)
   print("n=", n, " sum c l^alpha ", ms_sum, "sum u(x_l) ", ms_u_sum) 

for n in range(1,7):
    print("n=", n, " sum c l^alpha ", ms[n-1], "sum u(x_l) ", ms_u[n-1]) 


