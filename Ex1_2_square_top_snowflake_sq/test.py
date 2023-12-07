# Qile Yan 2023-12-7

# -div ( D grad u) = f in Omega
# u = g on bottom
# du/dn = kl / kr  on left and right
# Lambda du/dn + u = l on top
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
#  1: Top    x == 1
#  2: Right  y == 1
#  3: Bottom x == 0
#  4: Left  y == 0
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex1_solver import *
nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=int(input("Enter the degree of polynomial: "))

tolerance = 1e-7
max_iterations = 15
bc_top=1
bc_right=2
bc_bot=3
bc_left=4
geo = MakeGeometry(nn)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh = Mesh(ngmsh)

# Test 4: Domain is UnitSqaure with snow flake nn, solution is u = 2 + x^3 + 3xy
df=[]
mh=[]
err=[]
err2=[]
PETSc.Sys.Print("Test 4: Solution u=x^2+y+2 on UnitSquare")
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y = SpatialCoordinate(mesh)
  D = Constant(1.)
  Lambda = Constant(1.)
  f = -6*x
#  u = interpolate(2 + x**2 + y, FunctionSpace(mesh, "Lagrange",1))
  u = 2 + x**3 + x*y
  kl =-3*x**2-y
  kr = 3*x**2+y
  n = FacetNormal(mesh)
  l=inner(grad(u),n)+u
  uh = snowsolver(mesh, D, Lambda, f, u, kl, kr, l,deg,bc_right,bc_bot,bc_left,bc_top)
  V = FunctionSpace(mesh,"Lagrange",deg)
  mark, sum_eta = Mark(mesh, f,V,uh,tolerance)
  mesh = mesh.refine_marked_elements(mark)
  it=it+1
  meshplot = triplot(mesh)
  meshplot[0].set_linewidth(0.1)
  meshplot[1].set_linewidth(1)
  plt.xlim(-1, 2)
  plt.axis('equal')
  plt.title('Koch Snowflake Mesh')
  plt.savefig(f"refined_mesh/test_mesh/snow_{nn}_ref_{it}.pdf")
  plt.close()
  print("sum_eta is ", sum_eta)
  print(f"refined mesh plot saved to 'refined_mesh/test_mesh/snow_{nn}_ref_{it}.pdf'.")
  mh.append(mesh.cell_sizes.dat.data.max())
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Refined Mesh ", it, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in semi H1 norm is ", err2_temp)

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
plt.savefig(f"figures/koch_{nn}_test_dof.png")
PETSc.Sys.Print(f"Error vs degree of freedom  saved to figures/koch_{nn}_test_dof.png")
plt.close()
