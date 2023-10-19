import matplotlib.pyplot as plt
from firedrake import *
# choose a triangulation
nn=3
mesh_file = f'unit_cube_with_koch_n{nn}.msh'
mesh = Mesh(mesh_file)
MH = MeshHierarchy(mesh, 3)
mesh=MH[3]
n = FacetNormal(mesh)

# functoin space
V = FunctionSpace(mesh, "Lagrange", 1)

# define the exact solution and f
x, y = SpatialCoordinate(mesh)
D = 1
# Test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
Phi=[]
cc=[]
import numpy as np
LL = np.array([0.2,0.5,1,1.5,2,2.5,5,10,15,20])

for Lambda in LL:
  a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(6)
  L = Constant(0)*v*dx

# list of boundary  ids that corresponds to the exterior boundary of the domain
  boundary_ids = (5) # 
  bcs = DirichletBC(V, 1, boundary_ids)
  uh = Function(V)
  solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "cg", "pc_type": "none"})
  Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(5))
  Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(5))
  print(Phi_temp)
  print(Phi_temp2)
  Phi.append(Phi_temp)
  cc_temp=D*(13/9)**nn/Lambda
  print(cc_temp)
  cc.append(cc_temp)

fig, axes = plt.subplots()
plt.loglog(LL, Phi, marker='o')
plt.loglog(LL, cc,marker='o')
plt.legend(['$\Phi$', '$DL_p/\Lambda$'])
plt.xlabel('$\Lambda$')
plt.savefig("Phi_Lam_cube.png")
