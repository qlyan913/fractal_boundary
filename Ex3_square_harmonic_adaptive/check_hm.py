# Qile Yan 2024-02-20
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f=exp(-20*((x-0.5)**2+(y-0.5)**2))
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
max_iterations = 20
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
dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in range(5)]

N_all=[20,40,80,160,320,640,1280]
l_list=[] # size of segments
ms_list=[]
for N in N_all:
   l= (1./3.)**n/N
   l_list.append(l)
   ms=0
   alpha_list,c_list=get_alpha(uh,line_list,dy_list,N)[0:2]
   for i in range(len(alpha_list)):
     if alpha_list[i]>1:
        ms=ms+c_list[i]*l**alpha_list[i]
   ms_list.append(ms)

plt.figure()
plt.loglog(l_list,ms_list,'b.')
plt.xlabel('size of segments, $|l|$')
plt.legend([r'$\sum_{\alpha_l>1}c_l|l|^{\alpha_l}$'  ])
plt.savefig(f"figures/hm_n{n}.png")
plt.close()
print(f"plot of harmonic measure of edges  with alpha larger than 1 is saved in figures/hm_n{n}.png")



