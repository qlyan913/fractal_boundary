import matplotlib.pyplot as plt
from firedrake import *
# evaluate solution uh at several points
with CheckpointFile("solution.h5",'r') as afile:
  mesh=afile.load_mesh()
  uh=afile.load_function(mesh,"uh")

plt.figure()
ii=[1,2,3,4,5] 
uu=[] # value of solution at points
x_list=[]
pp1=[]
pp2=[]
for i in range(0,5):
   uu.append(uh.at([0.5,0.5*(3./2.)*(1-(1./3.)**(i+1))]))
   x_list.append(0.5*(3./2.)*(1-(1./3.)**(i+1)))
   pp1.append([0.5,0.5*(3./2.)*(1-(1./3.)**(i+1))])
   pp2.append([0.5*(3./2.)*(1-(1./3.)**(i+1)),0.5])
plt.plot(x_list,uu,marker='o')
plt.ylabel('evaluation of solution')
plt.xlabel('position of $x$')
plt.savefig("evaluate.png")
plt.close()

tt=x_list
tt=np.array([(x_list[1]/x_list[i])**3*uu[1] for i in range(0,len(uu))])
plt.figure()
plt.loglog(x_list,uu,marker='o')
plt.loglog(x_list,tt,marker='v')
plt.ylabel('evaluation of solution')
plt.xlabel('position of $x$')
plt.legend(['value of solution','$x^{-3}$'])
plt.savefig("evaluate_log.png")
plt.close()


print("evaluation of solution  at points:")
print(pp1)
print("value:")
print(uh.at(pp1))

print("evaluation of solution  at points: ")
print(pp2)
print("value:")
print(uh.at(pp2))

