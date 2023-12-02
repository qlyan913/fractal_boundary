# -div ( D grad u) = f in Omega
# u = 0 on bottom
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
# choose a triangulation
nn=3
#mesh_file = f'unit_cube_with_koch_n{nn}.msh'
#mesh = Mesh(mesh_file)
#MH = MeshHierarchy(mesh, 3)
#mesh=MH[3]
with CheckpointFile(f"refined_cube_3.h5","r") as afile:
     mesh=afile.load_mesh()
deg=1
n = FacetNormal(mesh)
# functoin space
V = FunctionSpace(mesh, "Lagrange", deg)

# define the exact solution and f
x, y,z = SpatialCoordinate(mesh)
D = 1
# Test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
Phi=[] # the flux through the top face 
cc=[]  # DL_p/\Lambda
import numpy as np
def get_flux(mesh,LL,nn,deg):
    for Lambda in LL:
        a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(6)
        L = Constant(0)*v*dx
        # list of boundary  ids that corresponds to the exterior boundary of the domain
        boundary_ids = (5) #
        bcs = DirichletBC(V, 1, boundary_ids)
        uh = Function(V)
        solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(6))
        Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(6))
        Phi.append(Phi_temp)
        cc_temp=D*(13/9)**nn/Lambda
        cc.append(cc_temp)
    return Phi,cc
LL = np.array([0.2,0.5,1,1.5,2,2.5,5,10,15,20,50,100,200,400,600,800,1000])
Phi, cc =get_flux(mesh0,LL,nn,deg)
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
plt.savefig("figures/Phi_Lam.png")
PETSc.Sys.Print(f"Plot of flux vs Lambda saved to figures/Phi_Lam_{nn}.png ")


# Region 1: Lambda <1
LL = np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.08,0.1,0.2,0.4,0.8,1])
Phi, cc =get_flux(mesh0,LL,nn,deg)
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
Phi, cc =get_flux(mesh0,LL,nn,deg)
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

