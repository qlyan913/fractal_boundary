import matplotlib.pyplot as plt
from firedrake import *
# evaluate solution uh at several points
with CheckpointFile("solution.h5",'r') as afile:
  mesh=afile.load_mesh()
  uh=afile.load_function(mesh,"uh")

plt.figure()
ii=[1,2,3,4,5]
uu=[]
for i in range(0,5):
   uu.append(uh.at([0.5,0.5*(3./2.)*(1-(1./3.)**(i+1))]))
plt.plot(ii,uu,marker='o')
plt.ylabel('evaluation of solution')
plt.xlabel('$i$-th point')
plt.savefig("evaluate.png")
