# -div ( D grad u) = 0 in Omega
# u = 1 on bottom
# du/dn= 0 on sides 
# Lambda du/dn+u=0 on top
#
#  Boundary surfaces are numbered as follows:
#  1: plane x == 0
#  2: plane x == 1
#  3: plane y == 0
#  4: plane y == 1
#  5: plane z == 0
#  6: plane z == 1 replaced by snowflake
#
# In this script, we calculate the flux through the top face, i.e., \int_top -D du/dn ds
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import csv
from netgen.occ import *
from geogen import *
from Ex2_solver import *
#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#deg=int(input("Enter the degree of polynomial: "))
nn=2
deg=1
tolerance = 1e-7
max_iterations = 2


# load the ngmesh
from netgen import meshing
ngmsh = meshing.Mesh(3) # create a 3-dimensional mesh
ngmsh.Load(f"domain/cube_{nn}.vol")


# Get label of boundary of Netgen mesh to index
bc_left = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['left']][0]
bc_right = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['right']][0]
bc_front = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['front']][0] 
bc_back = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['back']][0] 
bc_bot = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['bot']][0] 
bc_top = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['top']]
mesh0=Mesh(ngmsh)

mesh=mesh0

Phi=[] # the flux through the top face 
cc=[]  # DL_p/\Lambda
import numpy as np
def get_flux(mesh,LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top):
    for Lambda in LL:
        mesh=mesh0
        it=0
        sum_eta=1
        while sum_eta>tolerance and it<max_iterations:
           x, y,z = SpatialCoordinate(mesh)
           D = Constant(1.)
           f = Constant(0.)
           u_D=Constant(1.)
           k1 =Constant(0.)
           k2 =Constant(0.)
           k3 =Constant(0.)
           k4 =Constant(0.)
           n = FacetNormal(mesh)
           l=  Constant(0.)
           V = FunctionSpace(mesh,"Lagrange",deg)
           Lambda=1
           uh = snowsolver(mesh, D, Lambda, f, u_D, k1, k2,k3,k4, l,V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
           mark,sum_eta = Mark(mesh, f,uh,V,tolerance)
           PETSc.Sys.Print("error indicator sum_eta is ", sum_eta)
           #PETSc.Sys.Print("Refining the marked elements ...")
           Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(tuple(bc_top)))
           #Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(tuple(bc_top)))
           mesh = mesh.refine_marked_elements(mark)
           it=it+1
        Phi.append(Phi_temp)
        cc_temp=D*(13/9)**nn/Lambda
        cc.append(cc_temp)
    return Phi,cc  

LL = np.array([10**(-4),10**(-3),0.01,0.1,0.2,0.5,1,1.5,2,2.5,5,10,15,20,50,100,200,400,600,800,1000])
Phi, cc =get_flux(mesh0,LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
with open(f'results/Phi_Lam_{nn}.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux','DL_p/Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i], 'DL_p/Lambda':cc[i] })
 
PETSc.Sys.Print(f"Results saved to results/Phi_Lam_{nn}.csv")
fig, axes = plt.subplots()
plt.loglog(LL, Phi, marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}.png")
PETSc.Sys.Print(f"Plot of flux vs Lambda saved to figures/Phi_Lam_{nn}.png ")

# Region 1: Lambda <1 
LL = np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.08,0.1,0.2,0.4,0.8,1])
Phi, cc =get_flux(mesh0,LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
with open(f'results/Phi_Lam_{nn}_R1.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux','DL_p/Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i], 'DL_p/Lambda':cc[i] })
PETSc.Sys.Print(f"Results for 0<Lambda<1 saved to results/Phi_Lam_{nn}_R1.csv")  
  
fig, axes = plt.subplots()
plt.loglog(LL, Phi, marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R1.png")
PETSc.Sys.Print(f"Plot of flux vs Lambda  for 0<Lambda<1 saved to figures/Phi_Lam_{nn}_R1.png ")

# Region 2: 1<Lambda <L_p
Lp=(13/9)**nn
LL = np.linspace(1,Lp,15) 
Phi, cc =get_flux(mesh0,LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
with open(f'results/Phi_Lam_{nn}_R2.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux','DL_p/Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i], 'DL_p/Lambda':cc[i] })
PETSc.Sys.Print(f"Results for 1<Lambda<L_p saved to results/Phi_Lam_{nn}_R2.csv")  
  
fig, axes = plt.subplots()
plt.loglog(LL, Phi, marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R2.png")
PETSc.Sys.Print(f"Plot of flux vs Lambda  for 1<Lambda<L_p saved to figures/Phi_Lam_{nn}_R2.png ")
