# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f = exp(-20((x-0.5)^2+(x-0.5)^2))
#
# In this example, we would like to evaluate the solution along a random path of center of small squares.
#
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
#n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
#deg=int(input("Enter the degree of polynomial in FEM space:"))
file=open('input.txt','r')
n=int(file.readline())
deg=int(file.readline())
file.close()
import numpy as np
from firedrake import *
# choose a triangulation
mesh_file = f'domain/koch_{n}.msh'
mesh = Mesh(mesh_file)
# max of refinement
n_ref=3
MH=MeshHierarchy(mesh,n_ref)
# threshold for refinement in relative error
val_thr=10**(-3)
def snowsolver(mesh, f,g,V):
    # V = FunctionSpace(mesh, "Lagrange", 1)
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (1,2,3,4) # 1:top 2:right 3:bottom 4:left
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V,name="uh")
    #solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    return(uh)

# center points at center of squares of i-th iteration
pp=[[0.5,3/2-(1/3.)**i] for i in range(0,n+1)]
# distance to boundary
x_list=[(1/3.)**i for i in range(0,n+1)]
err=1
it=0
V = FunctionSpace(mesh, "Lagrange", deg)
x, y = SpatialCoordinate(mesh)
f=exp(-20*((x-0.5)**2+(y-0.5)**2))
g=0.0
uh0=snowsolver(mesh,f,g,V)

while err>val_thr and it<n_ref:
   it=it+1
   mesh=MH[it]
   x, y = SpatialCoordinate(mesh)
   V = FunctionSpace(mesh, "Lagrange", deg)
   f=exp(-20*((x-0.5)**2+(y-0.5)**2))
   g=0.0
   uh = snowsolver(mesh, f,g,V)
   uh0_c=Function(V)
   prolong(uh0,uh0_c)
   err=sqrt(assemble(dot(uh-uh0_c,uh-uh0_c)*dx))/sqrt(assemble(dot(uh,uh)*dx))
   PETSc.Sys.Print("The relative difference in L2 norm of solutions on coarse and fine mesh is", err)
   uh0=uh
 
del MH
# plot f
fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_{n}.png", dpi=1000)
PETSc.Sys.Print(f"The plot of force term f is saved to  figures/f_{n}.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png", dpi=1000)
PETSc.Sys.Print(f"The plot of solution is saved to figures/solution_{n}.png")

mesh.name="msh"
with CheckpointFile(f"solutions/solution_{n}.h5",'w') as afile:
  afile.save_mesh(mesh)
  afile.save_function(uh)
PETSc.Sys.Print(f"The mesh and solution is saved to solutions/solution_{n}.h5")

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
