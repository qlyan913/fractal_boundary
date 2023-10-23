# Qile Yan 2023-10-23
# Solve
#   -\Delta u =f in Omega
# with u = 0 on boundary
# f: equal to 1 on [1/3,2/3]x[1/3,2/3] and equal to 0 elsewhere
#
# In this example, we would like to evaluate the solution at center of small squares.
#
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
# choose a triangulation
mesh_file = 'unit_square_with_koch.msh'
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
    uh = Function(V)
    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    return(uh)

MH = MeshHierarchy(mesh, 5)
mesh=MH[5]
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 1)
f=conditional(And(And(And(1./3.<x,x<2./3.),1./3.<y),y<2./3.),1,0)
g=0
uh = snowsolver(mesh, f,g,V)

# plot f
fig, axes = plt.subplots()
ff=Function(V)
ff.interpolate(f)
collection = tripcolor(ff, axes=axes)
fig.colorbar(collection);
plt.savefig("f.png")

# plot solution
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig("solution.png")


