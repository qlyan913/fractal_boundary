import time
start_time = time.time()
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
from firedrake import *
# choose a triangulation
mesh_file = 'unit_square_with_koch.msh'
mesh = Mesh(mesh_file)
MH = MeshHierarchy(mesh, 3)
mesh=MH[3]
n = FacetNormal(mesh)

# functoin space
V = FunctionSpace(mesh, "Lagrange", 1)

# define the exact solution and f
x, y = SpatialCoordinate(mesh)
D = 1
nn= 4
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
  solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "cg", "pc_type": "none"})
  Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(4))
  Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(4))
  PETSc.Sys.Print("---Result with Lambda: %s  ---" % Lambda) 
  PETSc.Sys.Print("flux:",Phi_temp)
  PETSc.Sys.Print("flux computed by robin:",Phi_temp2)
  Phi.append(Phi_temp)
  cc_temp=D*(4/3)**nn/Lambda
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)
  cc.append(cc_temp)

PETSc.Sys.Print("--- running in %s seconds ---" % (time.time() - start_time))

fig, axes = plt.subplots()
plt.loglog(LL, Phi,marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig("Phi_Lam.png")
PETSc.Sys.Print("Result saved to Phi_Lam.png ")
