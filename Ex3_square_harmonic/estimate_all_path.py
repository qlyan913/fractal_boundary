import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
import numpy as np
import random
import statistics 
from scipy import stats
from firedrake import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
PETSc.Sys.Print("loading mesh")
with CheckpointFile(f"solutions/solution_{n}.h5",'r') as afile:
    mesh=afile.load_mesh('msh')
    uh=afile.load_function(mesh,"uh")

# distance to boundary
x_list=[(1/3.)**i for i in range(0,n+1)]

# step size 
h=[2*(1/3.)**(i+1) for i in range(0,n+1)] 

from itertools import product

def generate_lists_of_length_n(n):
    # Generate all possible combinations of 0, 1, and 2 for the given length n
    all_combinations = product([0, 1, 2], repeat=n)
    # Convert the combinations to lists
    result_lists = [list(combination) for combination in all_combinations]
    return result_lists
list_of_all_possible_number=generate_lists_of_length_n(n-1)

def all_path(n):
   # input: iterations for fractal
   # Output: sequence of points
   path_to_boundary=[]
   for nn in range(4):
      pp0=[np.array([0.5,0.5])]
      if nn==0:
          x10=np.array([0.5,0.5])+h[0]*np.array([1,0])
      elif nn==1:
          x10=np.array([0.5,0.5])+h[0]*np.array([0,1])
      elif nn==2:
          x10=np.array([0.5,0.5])+h[0]*np.array([-1,0])
      else:
          x10=np.array([0.5,0.5])+h[0]*np.array([0,-1])
      pp0.append(x10)
      R1=np.array([[1, 0],[0,1]])
      R2=np.array([[0 ,1],[1,0]])
      R3=np.array([[0 ,1],[-1,0]])
      for j in range(len(list_of_all_possible_number)):
         pp=pp0.copy()
         pj=list_of_all_possible_number[j]
         x0=np.array([0.5,0.5])
         x1=x10
         for i in range(1,n):
            d = (x1-x0)/np.linalg.norm(x1-x0)
            if pj[i-1]==0:
               d=np.matmul(R1,d)
            elif pj[i-1]==1:
               d=np.matmul(R2,d)
            else :
               d=np.matmul(R3,d)
            x2=x1+h[i]*d
            pp.append(x2)
            x0=x1
            x1=x2
         path_to_boundary.append(pp)
     
   return path_to_boundary

alpha0=[]
c0=[]
path_to_boundary=all_path(n)

for i in range(0,len(path_to_boundary)):
   uu=uh.at(path_to_boundary[i])
   # fitting log uu=log c+alpha log dx,
   x_list_log=np.log(x_list)
   uu_log=np.log(uu)
   #A = np.vstack([x_list_log, np.ones(len(x_list_log))]).T
   #beta=np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),uu_log)
   #c=exp(beta[1])
   #alpha=beta[0]
   res = stats.linregress(x_list_log, uu_log)
   c=exp(res.intercept)
   alpha=res.slope
   c0.append(c)
   alpha0.append(alpha)
   alpha=round(alpha,5)

alpha_mean=statistics.mean(alpha0)
alpha_std=statistics.stdev(alpha0)
c_mean=statistics.mean(c0)
c_std=statistics.stdev(c0)
PETSc.Sys.Print("Total number of the path from center to boundary is ", len(path_to_boundary), " which should be 4*3^(n-1)")
PETSc.Sys.Print("Mean of the alpha is % s, the standard deviation is %s " %(alpha_mean,alpha_std))
PETSc.Sys.Print("Mean of the c is % s, the standard deviation is %s " %(c_mean,c_std))

plt.hist(alpha0)
plt.savefig(f"figures/distribution_alpha.png")
plt.close()
PETSc.Sys.Print(f"plot of distribution of all estiamted alpha is saved in figures/distribution_alpha.png.")

plt.hist(c0)
plt.savefig(f"figures/distribution_c.png")
plt.close()
PETSc.Sys.Print(f"plot of distribution of all estiamted c is saved in figures/distribution_c.png.")










