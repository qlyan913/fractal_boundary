import matplotlib.pyplot as plt
import numpy as np
import random
from firedrake import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
trial=int(input("Enter the number of random paths to be estimated: "))
# Load the Gmsh mesh file
mesh_file = f'domain/koch_{n}.msh'
mesh = Mesh(mesh_file)

#meshplot = triplot(mesh)
#meshplot[0].set_linewidth(0.1)
#meshplot[1].set_linewidth(1)
#plt.xlim(-1, 2)
#plt.axis('equal')


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
   ppa=np.array(random_path(n))
   plt.plot(ppa[:,0],ppa[:,1],'r-')
   plt.savefig(f"figures/path_trial_{i}.pdf")
   plt.close()
   print(f"plot of path is saved in figures/path_trial_{i}.pdf.")








