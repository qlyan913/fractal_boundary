# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f: equal to 1 on [1/3,2/3]x[1/3,2/3] and equal to 0 elsewhere
#
# In this example, we would like to evaluate the solution at center of small squares.
#
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
deg=int(input("Enter the degree of polynomial in FEM space:"))
import numpy as np
from firedrake import *
# choose a triangulation
num_refinements = 4
mesh = UnitDiskMesh(num_refinements)
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

MH = MeshHierarchy(mesh, 3)
mesh=MH[3]
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", deg)
#f=conditional(And(And(And(1./3.<x,x<2./3.),1./3.<y),y<2./3.),1,0)
f=exp(-4*((x)**2+(y)**2))
g=0.0
uh = PDE_solver(mesh, f,g,V)

# plot f
fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_{n}.png")
PETSc.Sys.Print(f"The plot of force term f is saved to  figures/f_{n}.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png")
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_{n}.png")

with CheckpointFile(f"solutions/solution_{n}.h5",'w') as afile:
  afile.save_mesh(mesh)
  afile.save_function(uh)

# center points at center of squares of i-th iteration
pp=[[0.5,3/2-(1/3.)**i] for i in range(0,n+1)]
# distance to boundary
x_list=[(1/3.)**i for i in range(0,n+1)]

uu=uh.at(pp)
plt.figure()
plt.plot(x_list,uu,marker='o')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.savefig(f"figures/evaluate_{n}.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution is saved in figures/evaluate_{n}.png.")

tt=x_list
tt=np.array([(x_list[i]/x_list[2])**2.5*uu[2] for i in range(0,len(uu))])
plt.figure()
plt.loglog(x_list,uu,marker='o')
plt.loglog(x_list,tt,marker='v')
plt.ylabel('evaluation of solution')
plt.xlabel('distance to boundary')
plt.legend(['value of solution','$dist^{2.5}$'])
plt.savefig(f"figures/evaluate_log_{n}.png")
plt.close()
PETSc.Sys.Print(f"plot of evaluation of solution in loglog is saved in figures/evaluate_log_{n}.png.")

PETSc.Sys.Print("evaluation of solution  at points:")
PETSc.Sys.Print(pp)
PETSc.Sys.Print("value:")
PETSc.Sys.Print(uu)
