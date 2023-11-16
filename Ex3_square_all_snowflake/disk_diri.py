# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# Omega is unit disk
# f = exp(-20(x^2+y^2))
# We will evaluate the solution at (0,1-(1/2)^i), 0 <= i <= n_max
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from scipy import stats
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
val_thr=10**(-3)
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
V = FunctionSpace(mesh, "Lagrange", deg)
uh0=Function(V)
x, y = SpatialCoordinate(mesh)
uh0.interpolate(x)

while err>val_thr and it <n_ref:
   MH = MeshHierarchy(mesh, 1)
   mesh=MH[1]
   x, y = SpatialCoordinate(mesh)
   V = FunctionSpace(mesh, "Lagrange", deg)
   f=exp(-20*((x)**2+(y)**2))
   g=0.0
   uh = PDE_solver(mesh, f,g,V)
   uh0_c=Function(V)
   prolong(uh0,uh0_c)
   err=sqrt(assemble(dot(uh-uh0_c,uh-uh0_c)*dx))/sqrt(assemble(dot(uh,uh)*dx))
   uh0=uh
   it=it+1
 
 
if it == n_ref+1:
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
# fitting log uu=log c+alpha log dx, 
x_list_log=np.log(x_list)
uu_log=np.log(uu)
res = stats.linregress(x_list_log, uu_log)
c=exp(res.intercept)
alpha=res.slope
alpha=round(alpha,5)
tt=c*(x_list)**alpha
plt.figure()
plt.loglog(x_list,uu,'b.')
plt.loglog(x_list,tt)
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.legend(['value of solution','${%s}(dx)^{{%s}}$' % (round(c,5),alpha)])
plt.savefig(f"figures/evaluate_disk_log.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution in loglog is saved in figures/evaluate_disk_log.png.")

PETSc.Sys.Print("evaluation of solution  at points:")
PETSc.Sys.Print(pp)
PETSc.Sys.Print("value:")
PETSc.Sys.Print(uu)
