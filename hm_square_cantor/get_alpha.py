"""
Solve
   -\Delta u =0 in Omega
with u = 0 on Omega_int, u=1 on Omega_ext

Check the following:
    -\log u_n(x,3^-n)/(n\log 3) --> ? alpha(x,0)
"""
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
from hm_solver import *
from firedrake.pyplot import tripcolor
from matplotlib.ticker import PercentFormatter
nc=[1,2,3,4,5,6,7,8]
#deg=int(input("Enter the degree of polynomial in FEM space:"))
deg=5
mesh_size=2
u_all=[]
alpha_all=[]
# max of refinement
max_iterations = 40
# stop refinement when sum_eta less than tolerance
tolerance=1*1e-3
for n in nc:
   pp=[0,3**(-n)],[1./3.,3**(-n)],[2./3.,3**(-n)],[1,3**(-n)]
   # choose a triangulation
   geo = MakeGeometry(n)
   ngmsh = geo.GenerateMesh(maxh=mesh_size)
   mesh0 = Mesh(ngmsh)
   uh,f,V=get_solution(mesh0,tolerance,max_iterations,deg)
   u_val=uh.at(pp)
   u_all.append(u_val)
   alpha=-np.log(u_val)/(n*np.log(3))
   alpha_all.append(alpha)
 
with open(f'results/alpha.csv', 'w', newline='') as csvfile:
      fieldnames = ['x=0', 'x=1/3','x=2/3','x=1']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for i in range(len(alpha_all)):
         writer.writerow({'x=0': alpha_all[i][0], 'x=1/3':alpha_all[i][1],'x=2/3':alpha_all[i][2],'x=1':alpha_all[i][3]})
PETSc.Sys.Print(f"Results of alpha are saved to  results/alpha.csv")

with open(f'results/u_n.csv', 'w', newline='') as csvfile:
      fieldnames = ['x=0', 'x=1/3','x=2/3','x=1']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for i in range(len(alpha_all)):
         writer.writerow({'x=0': u_all[i][0], 'x=1/3':u_all[i][1],'x=2/3':u_all[i][2],'x=1':u_all[i][3]})
PETSc.Sys.Print(f"Evaluation of u are saved to  results/u_n.csv")
