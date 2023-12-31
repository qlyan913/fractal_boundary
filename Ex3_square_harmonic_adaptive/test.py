# Qile Yan 2023-10-21
sum_eta# Solve
#   -\Delta u =f in Omega
# u = g on boundary
#
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex3_solver import *

tolerance = 1e-7
max_iterations = 20

geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)

mesh = Mesh(ngmsh)

# Plot the mesh
print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
meshplot = triplot(mesh)
meshplot[0].set_linewidth(0.1)
meshplot[1].set_linewidth(1)
plt.xlim(-1, 2)
plt.axis('equal')
plt.title('Koch Snowflake Mesh')
plt.savefig(f"figures/snow_{n}.pdf")
plt.close()
print(f"Initial mesh plot saved to 'figures/snow_{n}.pdf'.")

# Test 1: Domain is UnitSqaure with snow flake n, solution is u = 2 + x + 3y
df=[]
err=[]
err2=[]
PETSc.Sys.Print("Test with solution  u=2+x^2+y ")
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y = SpatialCoordinate(mesh)
  f = Constant(-2.)
  u = 2 + x**2 + y
  V = FunctionSpace(mesh,"Lagrange",deg)
  uh = snowsolver(mesh, f,u,V)
  mark, sum_eta, eta_max = Mark(mesh, f,uh,V,tolerance)
  mesh = mesh.refine_marked_elements(mark)
  it=it+1
  meshplot = triplot(mesh)
  meshplot[0].set_linewidth(0.1)
  meshplot[1].set_linewidth(1)
  plt.xlim(-1, 2)
  plt.axis('equal')
  plt.title('Koch Snowflake Mesh')
  plt.savefig(f"refined_mesh/test_mesh/snow_{n}_ref_{it}.pdf")
  plt.close()
  print("sum_eta is ", sum_eta)
  print(f"refined mesh plot saved to 'refined_mesh/test_mesh/snow_{n}_ref_{it}.pdf'.")
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Refined Mesh ", it, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)
  
PETSc.Sys.Print(f"refined {it} times")
NN=np.array([(df[0]/df[i])**(1.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./2.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.loglog(df, NN)
plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(dof^{-1})$','$O(dof^{-1/2})$'])
plt.xlabel('degree of freedom')
plt.savefig(f"figures/koch_{n}_test_dof.png")
PETSc.Sys.Print(f"Error vs degree of freedom  saved to figures/koch_{n}_test_dof.png")
plt.close()



