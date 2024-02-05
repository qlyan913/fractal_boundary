import csv
import matplotlib.pyplot as plt
from firedrake import *
import statistics 
from scipy import stats
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex4_solver import *
import numpy as np
nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=int(input("Enter the degree of polynomial: "))
#mesh_size=1
#deg=5
#nn=2
l=(1/3)**nn
Lp=(2/3)**nn
dim_frac=np.log(2)/np.log(3)
tolerance = 1e-6
max_iterations = 20
bc_out=2
bc_int=1
D=1
geo = MakeGeometry(nn)

# get flux Phi0 at lambda = 0
Lambda=1
mesh_adap,uh,grad_uh=get_solution(geo,Lambda,D,mesh_size,tolerance,max_iterations,deg,bc_out,bc_int)
Phi0=get_flux(mesh_adap,uh,D,bc_int)
PETSc.Sys.Print("Lambda is ", Lambda, "flux is", Phi0)
file_name=f"results/solution_n{nn}_lambda_1.pvd"
export_to_pvd(file_name,mesh_adap,uh,grad_uh)
