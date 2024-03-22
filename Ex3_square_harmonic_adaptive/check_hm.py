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
x_list=[]
nv_list=[]

N=10
# divide the bottom edge into N segments
for L in line_list:
    pts_list,nv=divide_line_N(L,N)
    for pt in pts_list:
       x_list.append([pt,nv])

# estiamte alpha and c
alpha_list=[]
c_list=[]
pt_xlist=[]
pt_ylist=[]
std_list=[]
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
       pt_xlist.append(pt[0])
       pt_ylist.append(pt[1])
       std_list.append(res.stderr)


plt.figure()
plt.loglog(dy_list,uu,'b.')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.legend([r'$\sum_{\alpha_l>1}c_l|l|^{\alpha_l}$'  ])
plt.savefig(f"figures/hm_n{n}.png")
plt.close()
print(f"plot of harmonic measure of edges  with alpha larger than 1 is saved in figures/hm_n{n}.png")



