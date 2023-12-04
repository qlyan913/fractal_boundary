from firedrake.petsc import PETSc
import csv
import matplotlib.pyplot as plt
from firedrake import *
import statistics 
from scipy import stats
#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#deg=int(input("Enter the degree of polynomial: "))
nn=4
deg=3
l=(1/3)**nn
Lp=(4/3)**nn
# choose a triangulation
mesh_file =f'domain/unit_square_with_koch_{nn}.msh'
mesh = Mesh(mesh_file)
MH = MeshHierarchy(mesh, 5)
mesh=MH[5]
n = FacetNormal(mesh)

# functoin space
V = FunctionSpace(mesh, "Lagrange", deg)

# define the exact solution and f
x, y = SpatialCoordinate(mesh)
D = 1

# Test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# get flux Phi0 at lambda = 0
Lambda=10**(-8)
a = Constant(D)*dot(grad(u), grad(v))*dx++Constant(D)/Constant(Lambda)*u*v*ds(4)
L = Constant(0)*v*dx
# list of boundary  ids that corresponds to the exterior boundary of the domain
boundary_ids = (2) # 1: right; 2: bottom; 3:left; 4: top
bc1= DirichletBC(V, 1, boundary_ids)
uh = Function(V)
solve(a == L, uh, bcs=[bc1], solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre':'boomeramg'})
Phi0=assemble(Constant(D)/Constant(Lambda)*uh*ds(4))
print("phi0 is", Phi0)

# calculate flux for various Lambda 
Phi=[]
cc=[]
import numpy as np
#LL = np.array([10**(-5),0.0001,0.001,0.01,0.1,0.2,0.5,1,1.5,2,2.5,5,10,15,20,40,60,100,200,400,600,1000])
LL = np.array([2**i for i in range(-10,10)])
for Lambda in LL:
  a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(4)
  L = Constant(0)*v*dx
  # list of boundary  ids that corresponds to the exterior boundary of the domain
  boundary_ids = (2) # 1: right; 2: bottom; 3:left; 4: top
  bcs = DirichletBC(V, 1, boundary_ids)
  uh = Function(V)
  solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
  Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(4))
  Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(4))
  PETSc.Sys.Print("---Result with Lambda: %s  ---" % Lambda) 
  PETSc.Sys.Print("flux:",Phi_temp)
  PETSc.Sys.Print("flux computed by robin:",Phi_temp2)
  Phi.append(Phi_temp2)
  cc_temp=D*Lp/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
phi_2_log=np.log(phi_2)
#LL_log=np.log(LL)
#res = stats.linregress(phi_2_log, LL_log)
#c=exp(res.intercept)
alpha=1
plt.loglog(LL, phi_2,marker='o')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),marker='o')
plt.legend(['$1/\Phi-1/\Phi_0$', '$~\Lambda^{{%s}}$' % (alpha)])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}.png")
PETSc.Sys.Print(f"Plot for 0<Lambda<1000 saved to figures/Phi_Lam_{nn}.png ")
with open(f'results/Phi_Lam_{nn}.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux','DL_p/Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i], 'DL_p/Lambda':cc[i] })
PETSc.Sys.Print(f"Result for 0<Lambda<1000 saved to results/Phi_Lam_{nn}.csv ")


# Region 1: Lambda <l 
Phi=[]
cc=[]
LL = np.array([3**(-i) for i in range(nn,15)])
for Lambda in LL:
  a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(4)
  L = Constant(0)*v*dx
  # list of boundary  ids that corresponds to the exterior boundary of the domain
  boundary_ids = (2) # 1: right; 2: bottom; 3:left; 4: top
  bcs = DirichletBC(V, 1, boundary_ids)
  uh = Function(V)
  solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
  Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(4))
  Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(4))
  PETSc.Sys.Print("---Result with Lambda: %s  ---" % Lambda) 
  PETSc.Sys.Print("flux:",Phi_temp)
  PETSc.Sys.Print("flux computed by robin:",Phi_temp2)
  Phi.append(Phi_temp2)
  cc_temp=D*Lp/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
#phi_2_log=np.log(phi_2)
#LL_log=np.log(LL)
#res = stats.linregress(phi_2_log, LL_log)
#c=exp(res.intercept)
alpha=1
plt.loglog(LL, phi_2,marker='o')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),marker='o')
plt.legend(['$1/\Phi-1/\Phi_0$', '$~\Lambda^{{%s}}$' % (alpha)])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R1.png")
PETSc.Sys.Print(f"plot for 0<Lambda<l saved to figures/Phi_Lam_{nn}_R1.png ")
with open(f'results/Phi_Lam_{nn}_R1.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux','DL_p/Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i], 'DL_p/Lambda':cc[i] })
PETSc.Sys.Print(f"Result for 0<Lambda<\ell saved to results/Phi_Lam_{nn}_R1.csv ")

# Region 2: l< Lambda<L_p  
Phi=[]
cc=[]
LL = np.linspace(l,Lp,20) 
for Lambda in LL:
  a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(4)
  L = Constant(0)*v*dx
  # list of boundary  ids that corresponds to the exterior boundary of the domain
  boundary_ids = (2) # 1: right; 2: bottom; 3:left; 4: top
  bcs = DirichletBC(V, 1, boundary_ids)
  uh = Function(V)
  solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
  Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(4))
  Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(4))
  PETSc.Sys.Print("---Result with Lambda: %s  ---" % Lambda) 
  PETSc.Sys.Print("flux:",Phi_temp)
  PETSc.Sys.Print("flux computed by robin:",Phi_temp2)
  Phi.append(Phi_temp2)
  cc_temp=D*Lp/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
#phi_2_log=np.log(phi_2)
#LL_log=np.log(LL)
#res = stats.linregress(phi_2_log, LL_log)
#c=exp(res.intercept)
alpha=0.8
plt.loglog(LL, phi_2,marker='o')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),marker='o')
plt.legend(['$1/\Phi-1/\Phi_0$', '$~\Lambda^{{%s}}$' % (alpha)])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R2.png")
PETSc.Sys.Print(f"Plot for l<Lambda<L_p saved to figures/Phi_Lam_{nn}_R2.png ")
with open(f'results/Phi_Lam_{nn}_R2.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux','DL_p/Lambda']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i], 'DL_p/Lambda':cc[i] })
PETSc.Sys.Print(f"Result for l<Lambda<L_p saved to results/Phi_Lam_{nn}_R2.csv ")

