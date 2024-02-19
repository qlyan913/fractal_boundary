import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
import numpy as np
import random
import csv
import statistics 
from scipy import stats
from firedrake import *
from geogen import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
PETSc.Sys.Print("loading mesh")
with CheckpointFile(f"solutions/solution_{n}.h5",'r') as afile:
    mesh=afile.load_mesh('msh')
    uh=afile.load_function(mesh,"uh")
N= int(input("Enter the number of segments for estimation on the bottom  boundary: "))
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
for x in x_list:
    pt=x[0]
    nv=x[1]
    # distance to boundary
    dy_list=[0.5*(1/3.)**i for i in range(n,n+10)]
    # sequence of points
    pp=[pt-yy*nv for yy in dy_list]
    uu=uh.at(pp)
    dy_list_log=np.log(dy_list)
    uu_log=np.log(uu)
    res = stats.linregress(dy_list_log, uu_log)
    c=exp(res.intercept)
    alpha=res.slope
    alpha_list.append(alpha)
    c_list.append(c)

alpha_mean=statistics.mean(alpha_list)
alpha_std=statistics.stdev(alpha_list)
c_mean=statistics.mean(c_list)
c_std=statistics.stdev(c_list)
PETSc.Sys.Print("Total number of the points estimated on the bottom  boundary is ", len(x_list))
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
    fieldnames = ['alpha', 'c']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(alpha0)):
       writer.writerow({'alpha': alpha0[i], 'c':c0[i]})
PETSc.Sys.Print(f"Results of alpha and c are saved to  results/Results_n{n}_N{N}.csv")  








