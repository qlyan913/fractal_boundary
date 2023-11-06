import matplotlib.pyplot as plt
import numpy as np
import random
from firedrake import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
trial=int(input("Enter the number of random paths to be estimated: "))
print("loading mesh")
with CheckpointFile(f"solutions/solution_{n}.h5",'r') as afile:
    mesh=afile.load_mesh('msh')
    uh=afile.load_function(mesh,"uh")

# distance to boundary
x_list=[(1/3.)**i for i in range(0,n+1)]

# step size 
h=[2*(1/3.)**(i+1) for i in range(0,n+1)] 

def random_path(n):
   # Output: sequence of points 
   x0=np.array([0.5,0.5])
   pp=[np.array([0.5,0.5])]
   nn=random.randint(0,3)
   if nn==0:
       x1=np.array([0.5,0.5])+h[0]*np.array([1,0])
   elif nn==1:
       x1=np.array([0.5,0.5])+h[0]*np.array([0,1])
   elif nn==2:
       x1=np.array([0.5,0.5])+h[0]*np.array([-1,0])
   else:
       x1=np.array([0.5,0.5])+h[0]*np.array([0,-1])

   pp.append(x1)
   R1=np.array([[1, 0],[0,1]])
   R2=np.array([[0 ,1],[1,0]])
   R3=np.array([[0 ,1],[-1,0]])
   for i in range(1,n):
      d = (x1-x0)/np.linalg.norm(x1-x0)
      nn=random.randint(0,2)
      if nn==0:
         d=np.matmul(R1,d)
      elif nn==1:
         d=np.matmul(R2,d)
      else :
         d=np.matmul(R3,d)
      x2=x1+h[i]*d
      pp.append(x2)
      x0=x1
      x1=x2
   return pp

for i in range(0,trial):
   uu=uh.at(random_path(n)) 
   # fitting log uu=log c+alpha log dx, 
   x_list_log=np.log(x_list)
   A = np.vstack([x_list_log, np.ones(len(x_list_log))]).T
   uu_log=np.log(uu)
   beta=np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),uu_log)
   c=exp(beta[1])
   alpha=beta[0]
   alpha=round(alpha,5)
   tt=c*(x_list)**alpha
   plt.figure()
   plt.loglog(x_list,uu,'b.')
   plt.loglog(x_list,tt)
   plt.ylabel('evaluation of solution')
   plt.xlabel('distance to boundary')
   plt.legend(['value of solution','${%s}(dx)^{{%s}}$' % (round(c,5),alpha)])
   plt.savefig(f"figures/evaluate_{n}_trial_{i}.png")
   plt.close()
   print(f"plot of evaluation of solution in loglog is saved in figures/evaluate_{n}_trial_{i}.png.")








