# Qile Yan 2023-10-21
# Solve
#   -\Delta u =f in Omega
# u = g on boundary
#
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=int(intput("Enter the degree of polynomial in FEM space:"))
import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from firedrake.petsc import PETSc
from netgen.geom2d import SplineGeometry
from geogen import *
from slepc4py import SLEPc

tolerance = 1e-16
max_iterations = 10

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
PETSc.Sys.Print("Test with solution u=2+x^2+y")
it=0
for i in range(max_iterations):
  x, y = SpatialCoordinate(mesh)
  f = Constant(-2.)
  u = 2 + x**2 + y
  V = FunctionSpace(mesh,"Lagrange",deg)
  uh = snowsolver(mesh, f,u,V)
  mark = Mark(mesh, uh, f)
  mesh = mesh.refine_marked_elements(mark)
  it=it+1
  meshplot = triplot(mesh)
  meshplot[0].set_linewidth(0.1)
  meshplot[1].set_linewidth(1)
  plt.xlim(-1, 2)
  plt.axis('equal')
  plt.title('Koch Snowflake Mesh')
  plt.savefig(f"figures/snow_{n}_ref_{i}.pdf")
  plt.close()
  print(f"refined mesh plot saved to 'figures/snow_{n}_ref_{i}.pdf'.")
  df.append(V.dof_dset.layout_vec.getSize())
  err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
  err.append(err_temp)
  err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
  err2.append(err2_temp)
  PETSc.Sys.Print("Refined Mesh ", i, " with degree of freedom " , V.dof_dset.layout_vec.getSize())
  PETSc.Sys.Print("Error of solution in L2 norm is ", err_temp)
  PETSc.Sys.Print("Error of solution in H1 norm is ", err2_temp)
print(f"refined {it} times")
NN=np.array([(df[0]/df[i])**(1.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./2.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.legend(['$L^2$ error', '$H^1$ error'])
plt.xlabel('degree of freedom')
plt.savefig(f"figures/koch_{n}_test_dof.png")
PETSc.Sys.Print(f"Error vs degree of freedom  saved to figures/koch_{n}_test_dof.png")
plt.close()

def snowsolver(mesh, f,g,V):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (1,2,3,4) # 1:top 2:right 3:bottom 4:left
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V)
    #solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    return(uh)



def Mark(msh, uh, f):
     W = FunctionSpace(msh, "DG", 0)
     # Both the error indicator and the marked element vector will be DG0 field.
     w = TestFunction(W)
     R_T = f*uh + div(grad(uh))
     n = FacetNormal(V.mesh())
     h = CellDiameter(msh)
     R_dT = dot(grad(uh), n)
     # Assembling the error indicator.
     eta = assemble(h**2*R_T**2*w*dx +
           (h("+")+h("-"))*(R_dT("+")-R_dT("-"))**2*(w("+")+w("-"))*dS)
     frac = .95
     delfrac = .05
     part = .2
     mark = Function(W)
     # Filling in the marked element vector using eta.
     with mark.dat.vec as markedVec:
         with eta.dat.vec as etaVec:
             sum_eta = etaVec.sum()
             if sum_eta < tolerance:
                 return markedVec
             eta_max = etaVec.max()[1]
             sct, etaVec0 = PETSc.Scatter.toZero(etaVec)
             markedVec0 = etaVec0.duplicate()
             sct(etaVec, etaVec0)
             if etaVec.getComm().getRank() == 0:
                 eta = etaVec0.getArray()
                 marked = np.zeros(eta.size, dtype='bool')
                 sum_marked_eta = 0.
                 #Marking strategy
                 while sum_marked_eta < part*sum_eta:
                     new_marked = (~marked) & (eta > frac*eta_max)
                     sum_marked_eta += sum(eta[new_marked])
                     marked += new_marked
                     frac -= delfrac
                 markedVec0.getArray()[:] = 1.0*marked[:]
             sct(markedVec0, markedVec, mode=PETSc.Scatter.Mode.REVERSE)
     return mark
