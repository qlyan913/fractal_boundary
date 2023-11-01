# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f: equal to 1 on [1/3,2/3]x[1/3,2/3] and equal to 0 elsewhere
#
# In this example, we would like to evaluate the solution at center of small squares.
#
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
deg=int(input("Enter the degree of polynomial of FEM space:"))
import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
# choose a triangulation
mesh_file = f'koch_{n}.msh'
mesh = Mesh(mesh_file)
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
    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    return(uh)

MH = MeshHierarchy(mesh, 3)
mesh=MH[3]
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", deg)
f=conditional(And(And(And(1./3.<x,x<2./3.),1./3.<y),y<2./3.),1,0)
g=0.0
uh = snowsolver(mesh, f,g,V)

# plot f
fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/f_{n}.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/solution_{n}.png")

with CheckpointFile(f"solution_{n}.h5",'w') as afile:
  afile.save_mesh(mesh)
  afile.save_function(uh)

x_list=[3/2-(1/3.)**i for i in range(0,n+1)]
pp=[[0.5,3/2-(1/3.)**i] for i in range(0,n+1)]
uu=uh.at(pp)
plt.figure()
plt.plot(x_list,uu,marker='o')
plt.ylabel('evaluation of solution')
plt.xlabel('position of $x$')
plt.savefig(f"figures/evaluate_{n}.png")
plt.close()
print(f"plot of evaluation of solution is saved in figures/evaluate_{n}.png.")

tt=x_list
tt=np.array([(x_list[1]/x_list[i])**3*uu[1] for i in range(0,len(uu))])
plt.figure()
plt.loglog(x_list,uu,marker='o')
plt.loglog(x_list,tt,marker='v')
plt.ylabel('evaluation of solution')
plt.xlabel('position of $x$')
plt.legend(['value of solution','$x^{-3}$'])
plt.savefig(f"figures/evaluate_log_{n}.png")
plt.close()
print(f"plot of evaluation of solution in loglog is saved in figures/evaluate_log_{n}.png.")

print("evaluation of solution  at points:")
print(pp)
print("value:")
print(uu)
