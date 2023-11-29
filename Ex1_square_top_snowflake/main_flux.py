from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
from firedrake import *
#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#deg=int(input("Enter the degree of polynomial: "))
nn=4
deg=2
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
Phi=[]
cc=[]
import numpy as np
LL = np.array([0.2,0.5,1,1.5,2,2.5,5,10,15,20,40,60,100,200,400,600,1000])
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
  Phi.append(Phi_temp)
  cc_temp=D*Lp/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
plt.loglog(LL, Phi,marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig("figures/Phi_Lam.png")
PETSc.Sys.Print("Result for 0<Lambda<1000 saved to figures/Phi_Lam.png ")


# Region 1: Lambda <1 
Phi=[]
cc=[]
LL = np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.08,0.1,0.2,0.4,0.8,0.1])
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
  Phi.append(Phi_temp)
  cc_temp=D*Lp/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
plt.loglog(LL, Phi,marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig("figures/Phi_Lam_R1.png")
PETSc.Sys.Print("Result for 0<Lambda<1 saved to figures/Phi_Lam_R1.png ")

# Region 2: 1< Lambda<L_p  
Phi=[]
cc=[]
LL = np.linspace(1,Lp,15) 
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
  Phi.append(Phi_temp)
  cc_temp=D*Lp/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
plt.loglog(LL, Phi,marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig("figures/Phi_Lam_R2.png")
PETSc.Sys.Print("Result for 1<Lambda<L_p saved to figures/Phi_Lam_R2.png ")
