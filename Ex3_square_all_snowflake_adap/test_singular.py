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
tolerance = 1e-15
max_iterations = 20

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

ngmsh = geo.GenerateMesh(maxh=1)
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
df=[]
err=[]
err2=[]
#PETSc.Sys.Print("Test with solution  u=r^{2/3}sin(2/3 theta) ")
PETSc.Sys.Print("Test with solution  u=2+x^2+y ")
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y = SpatialCoordinate(mesh)
  f =Constant( 1.)
  u = Constant(0.) 
  #u=conditional(x>0,(x**2+y**2)**(1/3.)*sin(2/3*atan(y/x)),(x**2+y**2)**(1/3.)*sin(2/3*(atan(y/x)+pi)))
  V = FunctionSpace(mesh,"Lagrange",deg)
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
  df.append(V.dof_dset.layout_vec.getSize())
 # err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
 # err.append(err_temp)
 # err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx)) 
 # err2.append(err2_temp)
 # PETSc.Sys.Print("Refined Mesh ", it, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
 # PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
 # PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)


PETSc.Sys.Print(f"refined {it} times")

# Check the uniform refinement result
MH = MeshHierarchy(mesh_u, 8)
err_u=[]
err2_u=[]
df_u=[]
for i in range(0, len(MH)):
  mesh=MH[i]
  x, y = SpatialCoordinate(mesh)
  f = Constant(1.)
  u = Constant(0.)
  #u=conditional(x>0,(x**2+y**2)**(1/3.)*sin(2/3*atan(y/x)),(x**2+y**2)**(1/3.)*sin(2/3*(atan(y/x)+pi)))
  V = FunctionSpace(mesh,"Lagrange",deg)
  uh = snowsolver(mesh, f,u,V)
  meshplot = triplot(mesh)
  meshplot[0].set_linewidth(0.1)
  meshplot[1].set_linewidth(1)
  plt.xlim(0, 2)
  plt.axis('equal')
  plt.savefig(f"refined_mesh/test_singular_mesh/ref_uniform_{i}.pdf")
  plt.close()
  print(f"refined mesh plot saved to 'refined_mesh/test_singular_mesh/ref_uniform_{i}.pdf'.")
  df_u.append(V.dof_dset.layout_vec.getSize())
# err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  #err_u.append(err_temp)
 # err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
 # err2_u.append(err2_temp)
 # PETSc.Sys.Print("Refined Mesh ", i, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
 # PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)



#NN=np.array([(df_u[0]/df_u[i])**(1.)*err_u[0] for i in range(0,len(err_u))])
#NN2=np.array([(df_u[0]/df_u[i])**(1./2.)*err2_u[0] for i in range(0,len(err2_u))])
#plt.figure()
#plt.loglog(df, err,marker='o')
#plt.loglog(df, err2,marker='s')
#plt.loglog(df_u, err_u,marker='o')
#plt.loglog(df_u, err2_u,marker='s')
#plt.loglog(df_u, NN)
#plt.loglog(df_u, NN2)
#plt.legend(['$L^2$ error', '$H^1$ error','$L^2$ error-MeshH', '$H^1$ error-MeshH', '$O(dof^{-1})$','$O(dof^{-1/2})$'])
#plt.xlabel('degree of freedom')
#plt.savefig(f"figures/singular_test_dof.png")
#PETSc.Sys.Print(f"Error vs degree of freedom  saved to figures/singular_test_dof.png")
#plt.close()



