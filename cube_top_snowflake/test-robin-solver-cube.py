# Qile Yan 2023-09-29
#
# -div ( D grad u) = f in Omega
# u = g on bottom
# du/dn = ki,  ki is function defined on boundary 1<= i<= 4 
# Lambda du/dn + u = l on top
#
#
#  Weak formulation:
#
#  Find u in H1 with u = g on bottom such that
#
#    \int D grad(u).grad(v) dx  +  \int_{top} D/Lambda u v ds
#         = \int f v dx  +  \int_{top} (1/Lambda) l v ds
#           +  \int_{left} kl v ds   +  \int_{right} kr v ds
#
#  for all v in H1 which vanish on bottom.
#
#  Boundary surfaces are numbered as follows:
#  1: plane x == 0
#  2: plane x == 1
#  3: plane y == 0
#  4: plane y == 1
#  5: plane z == 0
#  6: plane z == 1 replaced by snowflake
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import time
def snowsolver(mesh, D, Lambda, f, g, k1, k2,k3,k4, l, V):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(6)
    L = f*v*dx + k1*v*ds(1) + k2*v*ds(2)+k3*v*ds(3) + k4*v*ds(4) + (1./Lambda)*l*v*ds(6)
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (5,)
    bcs = DirichletBC(V, g, boundary_ids)

    uh = Function(V)
    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    return(uh)

# Test 4: Domain is UnitSqaure with snow flake n=2, solution is u = 2 + x^2 + 3xy+y*z
n=3
mesh_file =f'unit_cube_with_koch_n{n}.msh'
mesh = Mesh(mesh_file)
df=[]
mh=[]
err=[]
err2=[]
PETSc.Sys.Print("Test: Solution u=2+x^2+3xy+yz on UnitCube")
PETSc.Sys.Print("Doing refinement ...")
start_time = time.time()
MH = MeshHierarchy(mesh, 3)
PETSc.Sys.Print("--- time used in doing refinement: %s seconds ---" % (time.time() - start_time))
for i in range(0, len(MH)):
  start_time = time.time()
  mesh=MH[i]
  x, y,z = SpatialCoordinate(mesh)
  D = Constant(1.)
  Lambda = Constant(1.)
  f = Constant(-2.)
  u = 2 + x**2 + 3*x*y+y*z
  u_D=2 + x**2 + 3*x*y
  k1 =-2*x-3*y
  k2 = 2*x+3*y
  k3 = -3*x-z
  k4 =3*x+z
  n = FacetNormal(mesh)
  l=inner(grad(u),n)+u
  V = FunctionSpace(mesh,"Lagrange",1)
  uh = snowsolver(mesh, D, Lambda, f, u, k1, k2,k3,k4, l,V) 
  mh.append(mesh.cell_sizes.dat.data.max())
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Refined Mesh ", i, " with max mesh size " , mesh.cell_sizes.dat.data.max())
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)
  PETSc.Sys.Print("Error of solution at bottom in L2 norm is ", sqrt(assemble(dot(uh - u_D, uh - u_D) * ds(5))))
  PETSc.Sys.Print("--- time used in solving PDE: %s seconds ---" % (time.time() - start_time))

NN=np.array([1.1*(mh[i]/mh[1])**2*err[1] for i in range(0,len(err))])
NN2=np.array([1.1*(mh[i]/mh[1])*err2[1] for i in range(0,len(err))])
plt.figure(2)
plt.loglog(mh, err,marker='o')
plt.loglog(mh, err2,marker='s')
plt.loglog(mh, NN)
plt.loglog(mh, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(h^2)$','$O(h)$'])
plt.xlabel('maximum of mesh size')
plt.title('Test 4')
plt.savefig("test_4.png")
PETSc.Sys.Print("Error vs mesh size  saved to test_4.png")
plt.close()

NN=np.array([(df[0]/df[i])**(2./3.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./3.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.loglog(df, NN)
plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(dof^{-2/3})$','$O(dof^{-1/3})$'])
plt.xlabel('degree of freedom')
plt.title('Test 4')
plt.savefig("test_4_dof.png")
PETSc.Sys.Print("Error vs mesh size  saved to test_4_dof.png")

