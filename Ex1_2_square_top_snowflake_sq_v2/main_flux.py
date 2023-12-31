from firedrake.petsc import PETSc
import csv
import matplotlib.pyplot as plt
from firedrake import *
import statistics 
from scipy import stats
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex1_solver import *
#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
#deg=int(input("Enter the degree of polynomial: "))
mesh_size=1
deg=5
nn=4
l=(1/5)**nn
Lp=(7/5)**nn
dim_frac=np.log(7)/np.log(5)
tolerance = 5*1e-8
max_iterations = 100
bc_top=(1)
bc_right=(2)
bc_bot=(3)
bc_left=(4)
geo = MakeGeometry(nn)
def get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top):
    Phi=[]
    for Lambda in LL:
        ngmsh = geo.GenerateMesh(maxh=mesh_size)
        mesh = Mesh(ngmsh)
        it = 0
        sum_eta=1
        while sum_eta>tolerance and it<max_iterations:
            x, y = SpatialCoordinate(mesh)
            n = FacetNormal(mesh)
            D = Constant(1.)
            f = Constant(0.)
            u_D=Constant(1.)
            kl=Constant(0.)
            kr=Constant(0.)
            l=Constant(0.)
            V = FunctionSpace(mesh,"Lagrange",deg)
            uh = snowsolver(mesh, D, Lambda, f, u_D, kl, kr, l,deg,bc_right,bc_bot,bc_left,bc_top)
     #       mark, sum_eta = Mark(mesh, f,V,uh,tolerance)
            mark,sum_eta=Mark_v2(mesh,Lambda, f, uh,V,tolerance,bc_left,bc_right,bc_top)
            PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize(), 'sum_eta is ', sum_eta)
            it=it+1
            Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(bc_top))
            Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(bc_top))
            mesh = mesh.refine_marked_elements(mark)
        PETSc.Sys.Print("Lambda is ", Lambda, " flux is ", Phi_temp2)
        Phi.append(Phi_temp2)
    return Phi
# get flux Phi0 at lambda = 0
Lambda=10**(-11)
Phi=get_flux(geo, [Lambda],nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
Phi0=Phi[0]
PETSc.Sys.Print("phi0 is", Phi0)

# calculate flux for various Lambda 
Phi=[]
import numpy as np
LL = np.array([2**i for i in range(-15,10)])
LL=np.append(LL,[l,Lp])
Phi=get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
phi_2_log=np.log(phi_2)
alpha=1/dim_frac
plt.loglog(LL[0:-2], phi_2[0:-2],marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[-1]**alpha)*(phi_2[-1]),color='red',linestyle='dashed',linewidth=0.8)
alpha=1
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='black',linestyle='dashed',linewidth=0.8)
plt.loglog(LL,(LL)**alpha/(LL[-3]**alpha)*(phi_2[-3]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.legend(['$1/\Phi-1/\Phi_0$','$O(\Lambda^{1/dim_frac})$', '$O(\Lambda^{1})$'])
plt.xticks([10**(-4),10**(-3),10**(-2),10**(-1),1,10**(1),10**(2),10**3,l, Lp], ['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','l', 'Lp'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}.png")
PETSc.Sys.Print(f"Plot for 0<Lambda<1000 saved to figures/Phi_Lam_{nn}.png ")
with open(f'results/Phi_Lam_{nn}.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i]})
PETSc.Sys.Print(f"Result for 0<Lambda<1000 saved to results/Phi_Lam_{nn}.csv ")


# Region 1: Lambda <l 
Phi=[]
LL = np.array([5**(-i) for i in range(nn,10)])
Phi=get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[-1]**alpha)*(phi_2[-1]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.xticks([10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),l],['$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$','$10^{-3}$','l'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R1.png")
PETSc.Sys.Print(f"plot for 0<Lambda<l saved to figures/Phi_Lam_{nn}_R1.png ")
with open(f'results/Phi_Lam_{nn}_R1.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i] })
PETSc.Sys.Print(f"Result for 0<Lambda<l saved to results/Phi_Lam_{nn}_R1.csv ")

# Region 2: l< Lambda<L_p  
Phi=[]
l_log=np.log(l)
Lp_log=np.log(Lp)
LL_log = np.linspace(l_log,Lp_log,20) 
LL=np.exp(LL_log)
Phi=get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1/dim_frac
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='red',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.xticks([10**(-2),10**(-1),1,l, Lp], ['$10^{-2}$','$10^{-1}$','$10^{0}$','l','Lp'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1/dim_frac})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R2.png")
PETSc.Sys.Print(f"Plot for l<Lambda<L_p saved to figures/Phi_Lam_{nn}_R2.png ")
with open(f'results/Phi_Lam_{nn}_R2.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i] })
PETSc.Sys.Print(f"Result for l<Lambda<L_p saved to results/Phi_Lam_{nn}_R2.csv ")


# Region 3: Lp< Lambda<infty
Phi=[]
LL=np.array([Lp*2**(i) for i in range(12)])
Phi=get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.xticks([10**(1),10**(2),10**(3),10**(4),10**(5), Lp], ['$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','Lp'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R3.png")
PETSc.Sys.Print(f"Plot for Lp<Lambda<infty saved to figures/Phi_Lam_{nn}_R3.png ")
with open(f'results/Phi_Lam_{nn}_R3.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i] })
PETSc.Sys.Print(f"Result for Lp<Lambda<infty saved to results/Phi_Lam_{nn}_R3.csv ")

