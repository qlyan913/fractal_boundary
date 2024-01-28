import csv
import matplotlib.pyplot as plt
from firedrake import *
import statistics 
from scipy import stats
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex1_solver import *
import numpy as np
#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#mesh_size=float(input("Enter the meshsize for initial mesh: "))
#deg=int(input("Enter the degree of polynomial: "))
mesh_size=1
deg=5
nn=2
l=(1/3)**nn
Lp=(4/3)**nn
dim_frac=np.log(4)/np.log(3)
tolerance = 1e-8
max_iterations = 40
bc_top=(1)
bc_right=(2)
bc_bot=(3)
bc_left=(4)
D=1
geo = MakeGeometry(nn)

# get flux Phi0 at lambda = 0
Lambda=10**(-11)
mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_right,bc_bot,bc_left,bc_top)
Phi0=get_flux(mesh_adap,uh,D,bc_top)
PETSc.Sys.Print("phi0 is", Phi0)
file_name=f"results/solution_{nn}.pvd"
export_to_pvd(file_name,mesh_adap,uh,grad_uh)

# calculate flux for various Lambda
Phi=[]
LL = np.array([2**i for i in range(-15,10)])
LL=np.append(LL,[l,Lp])
for Lambda in LL:
    mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_right,bc_bot,bc_left,bc_top)
    Phi0=get_flux(mesh_adap,uh,D,bc_top)
    PETSc.Sys.Print("Lambda is ",Lambda,"flux is", Phi0)
    Phi.append(Phi0)
    
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
LL = np.array([3**(-i) for i in range(nn,20)])
for Lambda in LL:
    mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_right,bc_bot,bc_left,bc_top)
    Phi0=get_flux(mesh_adap,uh,D,bc_top)
    PETSc.Sys.Print("Lambda is ",Lambda,"flux is", Phi0)
    Phi.append(Phi0)
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[-1]**alpha)*(phi_2[-1]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.xticks([10**(-9),10**(-8),10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),l],['$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$','$10^{-3}$','l'])
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
for Lambda in LL:
    mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_right,bc_bot,bc_left,bc_top)
    Phi0=get_flux(mesh_adap,uh,D,bc_top)
    PETSc.Sys.Print("Lambda is ",Lambda,"flux is", Phi0)
    Phi.append(Phi0)
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
LL=np.array([Lp*2**(i) for i in range(15)])
for Lambda in LL:
    mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_right,bc_bot,bc_left,bc_top)
    Phi0=get_flux(mesh_adap,uh,D,bc_top)
    PETSc.Sys.Print("Lambda is ",Lambda,"flux is", Phi0)
    Phi.append(Phi0)
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

