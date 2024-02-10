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
nn=5
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
#file_name=f"results/solution_{nn}.pvd"
#export_to_pvd(file_name,mesh_adap,uh,grad_uh)

# calculate flux for various Lambda
Phi=[]
LL = np.array([2**i for i in np.linspace(-25,14,40)])
for Lambda in LL:
    mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_right,bc_bot,bc_left,bc_top)
    Phi_temp=get_flux(mesh_adap,uh,D,bc_top)
    PETSc.Sys.Print("Lambda is ",Lambda,"flux is", Phi_temp)
    Phi.append(Phi_temp)
    
with open(f'results/Phi_Lam_{nn}_all.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i]})
PETSc.Sys.Print(f"Result for 0<Lambda<10000 saved to results/Phi_Lam_{nn}_all.csv ")


