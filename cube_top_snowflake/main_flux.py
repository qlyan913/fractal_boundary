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

n = FacetNormal(mesh)

# functoin space
V = FunctionSpace(mesh, "Lagrange", 1)

# define the exact solution and f
x, y,z = SpatialCoordinate(mesh)
D = 1
# Test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
Phi=[] # the flux through the top face 
cc=[]  # DL_p/\Lambda
import numpy as np
LL = np.array([0.2,0.5,1,1.5,2,2.5,5,10,15,20,50,100,200,400,600,800,1000])

for Lambda in LL:
  a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(6)
  L = Constant(0)*v*dx

# list of boundary  ids that corresponds to the exterior boundary of the domain
  boundary_ids = (5) # 
  bcs = DirichletBC(V, 1, boundary_ids)
  uh = Function(V)
  solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "cg", "pc_type": "none"})
  Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(6))
  Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(6))
  Phi.append(Phi_temp)
  cc_temp=D*(13/9)**nn/Lambda
  cc.append(cc_temp)
  PETSc.Sys.Print("---Result with Lambda: %s  ---" % Lambda)
  PETSc.Sys.Print("flux:",Phi_temp)
  PETSc.Sys.Print("flux computed by robin:",Phi_temp2)
  PETSc.Sys.Print("DL_p/Lambda:", cc_temp)

fig, axes = plt.subplots()
plt.loglog(LL, Phi, marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig("Phi_Lam_cube.png")
