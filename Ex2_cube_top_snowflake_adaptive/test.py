# Qile Yan 2023-10-21
#
# -div ( D grad u) = f in Omega
# u = g on bottom
# du/dn = ki,  ki is function defined on boundary 1<= i<= 4 
# Lambda du/dn + u = l on top
#
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
#
#  Boundary surfaces are numbered as follows:
#  1: plane x == 0
#  2: plane x == 1
#  3: plane y == 0
#  4: plane y == 1
#  5: plane z == 0
#  6: plane z == 1 replaced by snowflake
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import time
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex2_solver import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=input("Enter the degree of polynomial: ")

tolerance = 1e-7
max_iterations = 20
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)

# Plot the mesh
PETSc.Sys.Print(f'Finite element mesh has {mesh0.num_cells()} cells and {mesh0.num_vertices()} vertices.')
outfile = File(f"domain/cube_{n}.pvd")
outfile.write(mesh0)
PETSc.Sys.Print(f"File for visualization in Paraview saved as 'domain/cube_{n}.pvd.")

# Test : Domain is Unit Cube with snow flake n, solution is u = 2 + x^2 + 3xy+y*z
mesh = mesh0
df=[]
err=[]
err2=[]
PETSc.Sys.Print("Test: Solution u=2+x^2+3xy+yz on Unit Cube")
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y,z = SpatialCoordinate(mesh)
  D = Constant(1.)
  Lambda = Constant(1.)
  f = Constant(-2.)
  u = 2 + x**2 + 3*x*y+y*z
  u_D=2 + x**2 + 3*x*y
  k1 =-2*x-3*y
  k2 = 2*x+3*y
  k3 = -3*x-z
  k4 =3*x+z
  n = FacetNormal(mesh)
  l=inner(grad(u),n)+u
  V = FunctionSpace(mesh,"Lagrange",deg)
  PETSc.Sys.Print("Solving the PDE ...")
  uh = snowsolver(mesh, D, Lambda, f, u, k1, k2,k3,k4, l,V)
  mark,sum_eta = Mark(mesh, f,uh,V,tolerance)
  mesh = mesh.refine_marked_elements(mark)
  it=it+1
  outfile = File(f"refined_mesh/test_mesh/ref_n{n}_{it}.pvd")
  outfile.write(mesh)
  PETSc.Sys.Print(f"Refined mesh saved as 'refined_mesh/test_mesh/ref_n{n}_{it}.pvd.")
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)
  PETSc.Sys.Print("Error of solution at bottom in L2 norm is ", sqrt(assemble(dot(uh - u_D, uh - u_D) * ds(5))))

NN=np.array([(df[0]/df[i])**(2./3.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./3.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.loglog(df, NN)
plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(dof^{-2/3})$','$O(dof^{-1/3})$'])
plt.xlabel('degree of freedom')
plt.title('Test with soluion $u=2+x^2+3xy+yz$')
plt.savefig("figures/test_dof.png")
PETSc.Sys.Print("Error vs mesh size  saved to figures/test_dof.png")


