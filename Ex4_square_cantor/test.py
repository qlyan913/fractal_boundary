# Qile Yan 2024-02-01
#
# -div ( D grad u) = f in Omega
# The domain is Omega=Q\Q_n where Q=[-1,1] x [-1, 1] and Q_n is the nth iteration of 2D cantor set （cantor dust) inside Q.
# u = g on \partial Q
# Lambda du/dn + u = l on \partial Q_n
#
#  Weak formulation:
#
#  Find u in H1 with u = g on bottom such that
#
#    \int D grad(u).grad(v) dx  +  \int_{\partial Q_n} D/Lambda u v ds
#         = \int f v dx  +  \int_{\partial Q_n} (1/Lambda) l v ds
#  for all v in H1 which vanish on bottom.
#  Index of boundary
#  1: Outside boundary
#  2: Inside boundary
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex4_solver import *
nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=int(input("Enter the degree of polynomial: "))
tolerance = 1e-5
max_iterations = 15
bc_out=1
bc_int=2
geo = MakeGeometry(nn)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh = Mesh(ngmsh)

# Test 4: Domain is UnitSqaure with snow flake nn, solution is u = 2 + x^3 + 3xy
df=[]
mh=[]
err=[]
err2=[]
PETSc.Sys.Print("Test 4: Solution u=x^2+y+2 on Domain with Cantor dust")
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y = SpatialCoordinate(mesh)
  D = Constant(1.)
  Lambda = Constant(1.)
  f = -30*x**4
#  u = interpolate(2 + x**2 + y, FunctionSpace(mesh, "Lagrange",1))
  u = 2 + x**6 + x*y
  kl =-6*x**5-y
  kr = 6*x**5+y
  n = FacetNormal(mesh)
  l=inner(grad(u),n)+u
  uh = snowsolver(mesh, D, Lambda, f, u, l,deg,bc_out,bc_int)
  V = FunctionSpace(mesh,"Lagrange",deg)
  mark, sum_eta = Mark(mesh, f,V,uh,tolerance)
#  mark,sum_eta=Mark_v2(mesh,Lambda, f, uh,V,tolerance,bc_left,bc_right,bc_top)
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
