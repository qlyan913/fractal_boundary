# Qile Yan 2023-10-21
# Solve
#   -\Delta u =f in Omega
# u = g on boundary
# Omega is a singular domain:
# -------------------------
# |                        |
# |                        |
# |                        |
# |            ------------
# |           |
# |           |
# |           |
# |------------
#  1: Top    x == 1
#  2: Right  y == 1
#  3: Bottom x == 0
#  4: Left   y == 0
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex3_solver import *
deg=int(input("Enter the degree of polynomial in FEM space:"))
tolerance = 1e-5
max_iterations = 12

geo = SplineGeometry()
p1=geo.AppendPoint(*[-1,-1])
p2=geo.AppendPoint(*[0,-1])
p3=geo.AppendPoint(*[0,0])
p4=geo.AppendPoint(*[1,0])
p5=geo.AppendPoint(*[1,1])
p6=geo.AppendPoint(*[-1,1])
geo.Append (["line",p1 ,p2], bc =3 )
geo.Append (["line",p2 ,p3], bc =2 )
geo.Append (["line",p3 ,p4], bc =2 )
geo.Append (["line",p4 ,p5], bc =2 )
geo.Append (["line",p5 ,p6], bc =1 )
geo.Append (["line",p6 ,p1], bc =4 )

ngmsh = geo.GenerateMesh(maxh=0.5)
mesh = Mesh(ngmsh)
mesh_u=mesh
# Plot the mesh
print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
meshplot = triplot(mesh)
meshplot[0].set_linewidth(0.1)
meshplot[1].set_linewidth(1)
plt.xlim(0, 2)
plt.axis('equal')
plt.title('Koch Snowflake Mesh')
plt.savefig(f"figures/singular.pdf")
plt.close()
print(f"Initial mesh plot saved to 'figures/singular.pdf'.")

# Test 1: Domain is UnitSqaure with snow flake n, solution is u = 2 + x^2 + y

#PETSc.Sys.Print("Test with solution  u=r^{2/3}sin(2/3 theta) ")
PETSc.Sys.Print("Test with solution f=1 in omega, u=0 at boundary ")
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y = SpatialCoordinate(mesh)
  V = FunctionSpace(mesh,"Lagrange",deg) 
  u=Constant(0.)
  f=Constant(1.)
  uh = snowsolver(mesh, f,u,V)
  mark,sum_eta = Mark(mesh, f,uh,V,tolerance)
  mesh = mesh.refine_marked_elements(mark)
  it=it+1
  meshplot = triplot(mesh)
  meshplot[0].set_linewidth(0.1)
  meshplot[1].set_linewidth(1)
  plt.xlim(0, 2)
  plt.axis('equal')
  plt.savefig(f"refined_mesh/test_singular_mesh/ref_{it}.pdf")
  plt.close()
  print(f"refined mesh plot saved to 'refined_mesh/test_singular_mesh/ref_{it}.pdf'.")
  PETSc.Sys.Print("Refined Mesh ", it, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
  

PETSc.Sys.Print(f"refined {it} times")

fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/soln_adap.png")
PETSc.Sys.Print(f"The plot of solution is saved to figures/soln_adap.png")
plt.close()

# Check the uniform refinement result
MH = MeshHierarchy(mesh_u, 8)

for i in range(0, len(MH)):
  mesh=MH[i]
  x, y = SpatialCoordinate(mesh)
  V = FunctionSpace(mesh,"Lagrange",deg)
  u=Constant(0.)
  f=Constant(1.)
  uh = snowsolver(mesh, f,u,V)
  meshplot = triplot(mesh)
  meshplot[0].set_linewidth(0.1)
  meshplot[1].set_linewidth(1)
  plt.xlim(0, 2)
  plt.axis('equal')
  plt.savefig(f"refined_mesh/test_singular_mesh/ref_uniform_{i}.pdf")
  plt.close()
  print(f"refined mesh plot saved to 'refined_mesh/test_singular_mesh/ref_uniform_{i}.pdf'.")
  PETSc.Sys.Print("Refined Mesh ", i, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
  
fig, axes = plt.subplots()
collection = tripcolor(uh, axes=axes)
fig.colorbar(collection);
plt.savefig(f"figures/soln_uniform.png")
PETSc.Sys.Print(f"The plot of solution is saved to figures/soln_uniform.png")
plt.close()



