# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# Omega is unit disk
# f: equal to 1 on [1/3,2/3]x[1/3,2/3] and equal to 0 elsewhere
# or
# f = exp(-4(x^2+y^2))
# We will evaluate the solution at (0,1-(1/2)^i), 0 <= i <= n_max
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
deg=int(input("Enter the degree of polynomial in FEM space:"))
import numpy as np
from firedrake import *
# choose a triangulation
num_refinements = 4
mesh = UnitDiskMesh(num_refinements)
# number of points to be evaluated
n_max=10 
# max of refinement
n_ref=5
# threshold of refinement for relative error 
val_thr=10**(-2)
def PDE_solver(mesh, f,g,V):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (1) # 
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V,name="uh")
    #solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    return(uh)

# points goes to boundary of unit disk
pp=[[0,1-(1/2.)**i] for i in range(0,n_max)]
# distance to boundary
x_list=[(1/2.)**i for i in range(0,n_max)]
err=1
it=0
uu0=np.ones(n_max)
while err>val_thr and it <n_ref:
   MH = MeshHierarchy(mesh, 1)
   mesh=MH[1]
   x, y = SpatialCoordinate(mesh)
   V = FunctionSpace(mesh, "Lagrange", deg)
   #f=conditional(And(And(And(1./3.<x,x<2./3.),1./3.<y),y<2./3.),1,0)
   f=exp(-4*((x)**2+(y)**2))
   g=0.0
   uh = PDE_solver(mesh, f,g,V)
   uu = np.array(uh.at(pp))
   err=np.linalg.norm((uu-uu0)/uu0,np.inf)
   uu0=uu
   it=it+1
 
if it == n_ref:
   PETSc.Sys.Print("maximum number of refinement is reached")

# plot f
fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_disk.png")
PETSc.Sys.Print(f"The plot of force term f is saved to  figures/f_disk.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_disk.png")
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_disk.png")

uu=uh.at(pp)
plt.figure()
plt.plot(x_list,uu,marker='o')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.savefig(f"figures/evaluate_disk.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution is saved in figures/evaluate_disk.png.")

tt=x_list
tt=np.array([(x_list[i]/x_list[2])*uu[2] for i in range(0,len(uu))])
plt.figure()
plt.loglog(x_list,uu,marker='o')
plt.loglog(x_list,tt,marker='v')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.legend(['value of solution','$dist^{1}$'])
plt.savefig(f"figures/evaluate_disk_log.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution in loglog is saved in figures/evaluate_disk_log.png.")

PETSc.Sys.Print("evaluation of solution  at points:")
PETSc.Sys.Print(pp)
PETSc.Sys.Print("value:")
PETSc.Sys.Print(uu)
