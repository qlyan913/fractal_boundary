# Qile Yan 2023-10-21
# Solve
#   -\Delta u =f in Omega
# u = g on boundary
#
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
def snowsolver(mesh, f,g):
    V = FunctionSpace(mesh, "Lagrange", 1)
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (1,2,3,4) # 1:top 2:right 3:bottom 4:left
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V)
    #solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'}) 
    return(uh)

# Test 1: Domain is UnitSqaure with snow flake n, solution is u = 2 + x + 3y
mesh_file = f'domain/koch_{n}.msh'
mesh = Mesh(mesh_file)
df=[]
mh=[]
err=[]
err2=[]
PETSc.Sys.Print("Test with solution u=2+x+3y on Unit Square")
PETSc.Sys.Print("Doing refinement ...")
MH = MeshHierarchy(mesh, 5)
for i in range(0, len(MH)):
  mesh=MH[i]
  x, y = SpatialCoordinate(mesh)
  f = Constant(0.)
  u = 2 + x +3*y
  uh = snowsolver(mesh, f,u)
  V = FunctionSpace(mesh,"Lagrange",1)
  mh.append(mesh.cell_sizes.dat.data.max())
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Refined Mesh ", i, " with max mesh size " , mesh.cell_sizes.dat.data.max())
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)

NN=np.array([1.1*(mh[i]/mh[1])**2*err[1] for i in range(0,len(err))])
NN2=np.array([1.1*(mh[i]/mh[1])*err2[1] for i in range(0,len(err))])
plt.figure()
plt.loglog(mh, err,marker='o')
plt.loglog(mh, err2,marker='s')
#plt.loglog(mh, NN)
#plt.loglog(mh, NN2)
plt.legend(['$L^2$ error', '$H^1$ error'])
plt.xlabel('maximum of mesh size')
#plt.title('Test')
plt.savefig(f"figures/koch_{n}_test1.png")
PETSc.Sys.Print(f"Error vs mesh size  saved to figures/koch_{n}_test1.png")
plt.close()

NN=np.array([(df[0]/df[i])**(1.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./2.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
#plt.loglog(df, NN)
#plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error'])
plt.xlabel('degree of freedom')
#plt.title('Test')
plt.savefig(f"figures/koch_{n}_test1_dof.png")
PETSc.Sys.Print(f"Error vs degree of freedom  saved to figures/koch_{n}_test1_dof.png")
plt.close()

# Test 2: Domain is UnitSqaure with snow flake n, solution is u = 2 + x^2 + 3xy
df=[]
mh=[]
err=[]
err2=[]
PETSc.Sys.Print("Test with solution u=2+x^2+y+2 on UnitSquare")
for i in range(0, len(MH)):
  mesh=MH[i]
  x, y = SpatialCoordinate(mesh)
  f = Constant(-2.)
  u = 2 + x**2 + y
  uh = snowsolver(mesh, f,u)
  V = FunctionSpace(mesh,"Lagrange",1)
  mh.append(mesh.cell_sizes.dat.data.max())
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Refined Mesh ", i, " with max mesh size " , mesh.cell_sizes.dat.data.max())
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)

NN=np.array([1.1*(mh[i]/mh[1])**2*err[1] for i in range(0,len(err))])
NN2=np.array([1.1*(mh[i]/mh[1])*err2[1] for i in range(0,len(err))])
plt.figure()
plt.loglog(mh, err,marker='o')
plt.loglog(mh, err2,marker='s')
plt.loglog(mh, NN)
plt.loglog(mh, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(h^2)$','$O(h)$'])
plt.xlabel('maximum of mesh size')
plt.savefig(f"figures/koch_{n}_test2.png")
PETSc.Sys.Print(f"Error vs mesh size  saved to figures/koch_{n}_test2.png")
plt.close()

NN=np.array([(df[0]/df[i])**(1.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./2.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.loglog(df, NN)
plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(dof^{-1})$','$O(dof^{-1/2})$'])
plt.xlabel('degree of freedom')
plt.savefig(f"figures/koch_{n}_test2_dof.png")
PETSc.Sys.Print(f"Error vs degree of freedom  saved to figures/koch_{n}_test2_dof.png")
