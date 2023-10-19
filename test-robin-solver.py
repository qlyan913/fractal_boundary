# Qile Yan 2023-09-21

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

import matplotlib.pyplot as plt
from firedrake import *
import numpy as np

def snowsolver(mesh, D, Lambda, f, g, kl, kr, l):
    V = FunctionSpace(mesh, "Lagrange", 1)
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(4)
    L = f*v*dx + kl*v*ds(3) + kr*v*ds(1) + (1./Lambda)*l*v*ds(4)
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (2,) # 1: right; 2: bottom; 3:left; 4: top
    bcs = DirichletBC(V, g, boundary_ids)

    uh = Function(V)
    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    return(uh)

# Test 1.  exact solution is u = 1 (constant)
# choose a precomputed triangulation
# mesh boundary should be marked: 1: right; 2: bottom; 3:left; 4: top
mesh_file = 'unit_square_with_koch.msh'
mesh = Mesh(mesh_file)
x, y = SpatialCoordinate(mesh)
D = Constant(1.)
Lambda = Constant(1.)
f = Constant(0.)
u= Constant(1.)
kl = Constant(0.)
kr = Constant(0.)
l = Constant(1.)

uh = snowsolver(mesh, D, Lambda, f, u, kl, kr, l)
print("Test 1: Solution u==1 on unit sqaure with snow flake ")
print("Error in L2 norm is ", sqrt(assemble(dot(uh - u, uh - u) * dx)))
#print(uh.vector().array())
#print("answer should be [1. 1. 1. 1. 1. 1. 1. 1.]")

# Test 2.  Domain is a trapezoid, solution is u = 2 + x + 3 y
mesh_file = 'trapezoid.msh'
mesh = Mesh(mesh_file)
x, y = SpatialCoordinate(mesh)
D = Constant(1.)
Lambda = Constant(1.)
f = Constant(0.)
u = interpolate(2 + x + 3*y, FunctionSpace(mesh, "Lagrange",1))
kl = Constant(-1.)
kr = Constant(1.)
l = Constant(np.sqrt(2.))+u
uh = snowsolver(mesh, D, Lambda, f, u, kl, kr, l)

print("Test 2: Solution u=2+x+3y on a trapezoid ")
print("Error in L2 norm is ", sqrt(assemble(dot(uh - u, uh - u) * dx)))

# Test 2-2. Use FacetNormal to get normal vector at boundary and define the robin boundary condition
n = FacetNormal(mesh)
l2=inner(grad(u),n)+u
uh = snowsolver(mesh, D, Lambda, f, u, kl, kr, l2)
print("Test 2-2: Solution u=2+x+3y on a trapezoid, use FacetNoraml to define normal derivatives in robin boundary condition ")
print("Error of normal derivatives on top  is ", sqrt(assemble(dot(l - l2, l - l2) * ds(4))))
print("Error of solution in L2 norm is ", sqrt(assemble(dot(uh - u, uh - u) * dx)))

# Test 3: Domain is a trapezoid, solution is u = 2 + x^2 + 3xy
mesh_file = 'trapezoid.msh'
mesh = Mesh(mesh_file)
MH = MeshHierarchy(mesh, 5)
df=[] # mesh size/ degree of freedom
err=[] # error of solution
print("Test 3: Solution u=2+x^2+3xy on a trapezoid")
for i in range(0,len(MH)):
  mesh = MH[i]
  x, y = SpatialCoordinate(mesh)
  D = Constant(1.)
  Lambda = Constant(1.)
  f = Constant(-2.)
#  u = interpolate(2 + x**2 + 3*x*y, FunctionSpace(mesh, "Lagrange",1))
  u = 2 + x**2 + 3*x*y 
  kl =-2*x-3*y
  kr = 2*x+3*y
  n = FacetNormal(mesh)
  l2=inner(grad(u),n)+u
  l = -0.5*np.sqrt(2)*(2*x+3*y)+0.5*np.sqrt(2)*3*x+u
  uh = snowsolver(mesh, D, Lambda, f, u, kl, kr, l2)
  V = FunctionSpace(mesh,"Lagrange",1)
#  df.append(V.dof_dset.layout_vec.getSize())
#  df.append(mesh.num_cells())
  df.append(mesh.cell_sizes.dat.data.max())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  print("Refined Mesh ", i," with max mesh size ", mesh.cell_sizes.dat.data.max())
  print("Error of normal derivatives on top  is ", sqrt(assemble(dot(l - l2, l - l2) * ds(4))))
  print("Error of solution in L2 norm is ", err_temp)
import math
NN=np.array([1.1*(df[i]/df[0])**2*err[0] for i in range(0,len(err))])
plt.figure(1)
plt.loglog(df, err)
plt.loglog(df, NN)
plt.legend(['error', '$O(h^2)$'])
plt.xlabel('maximum of mesh size')
plt.title('Test 3')
plt.savefig("test_3.png")
print("Error vs mesh size saved to test_3.png")
plt.close()

# Test 4: Domain is UnitSqaure with snow flake n=4, solution is u = 2 + x^2 + 3xy
mesh_file = 'unit_square_with_koch.msh'
mesh = Mesh(mesh_file)
df=[]
mh=[]
err=[]
err2=[]
print("Test 4: Solution u=x^2+y+2 on UnitSquare")
print("Doing refinement ...")
MH = MeshHierarchy(mesh, 4)
for i in range(0, len(MH)):
  mesh=MH[i]
  x, y = SpatialCoordinate(mesh)
  D = Constant(1.)
  Lambda = Constant(1.)
  f = Constant(-2.)
#  u = interpolate(2 + x**2 + y, FunctionSpace(mesh, "Lagrange",1))
  u = 2 + x**2 + y
  kl =-2*x
  kr = 2*x
  n = FacetNormal(mesh)
  l=inner(grad(u),n)+u
  uh = snowsolver(mesh, D, Lambda, f, u, kl, kr, l)
  V = FunctionSpace(mesh,"Lagrange",1)
  mh.append(mesh.cell_sizes.dat.data.max())
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  print("Refined Mesh ", i, " with max mesh size " , mesh.cell_sizes.dat.data.max())
  print("Error of solution in L2 norm is ", err_temp)
  print("Error of solution in H1 norm is ", err2_temp)

NN=np.array([1.1*(mh[i]/mh[1])**2*err[1] for i in range(0,len(err))])
NN2=np.array([1.1*(mh[i]/mh[1])*err2[1] for i in range(0,len(err))])
plt.figure(2)
plt.loglog(mh, err,marker='o')
plt.loglog(mh, err2,marker='s')
plt.loglog(mh, NN)
plt.loglog(mh, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(h^2)$','$O(h)$'])
plt.xlabel('maximum of mesh size')
plt.title('Test 4')
plt.savefig("test_4.png")
print("Error vs mesh size  saved to test_4.png")
plt.close()

NN=np.array([(df[0]/df[i])**(1.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./2.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.loglog(df, NN)
plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(dof^{-1})$','$O(dof^{-1/2})$'])
plt.xlabel('degree of freedom')
plt.title('Test 4')
plt.savefig("test_4_dof.png")
print("Error vs mesh size  saved to test_4_dof.png")
