# solve -div(grad u)=f with u=0 on boundary
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
from netgen.occ import *
# degree of polynomial
deg = 3
# parameter in adaptive meshing
tolerance = 1e-8
max_iterations = 100
# create netgen mesh
cube1 =  Box(Pnt(0,0,0), Pnt(2,2,2))
cube1.bc('DC')
cube2 = Box(Pnt(1,0,1),Pnt(2,1,2))
cube2.bc('DC')
domain=cube1-cube2
geo = OCCGeometry(domain)
ngmsh = geo.GenerateMesh(maxh=0.2)
DC_inds = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['DC']]

mesh0=Mesh(ngmsh)
# Plot the mesh
PETSc.Sys.Print(f'Finite element mesh has {mesh0.num_cells()} cells and {mesh0.num_vertices()} vertices.')
outfile = File(f"cube_example.pvd")
outfile.write(mesh0)
PETSc.Sys.Print(f"File for visualization in Paraview saved as 'cube_example.pvd.")

def Esolver(mesh,f,V,DC_inds):
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    boundary_ids = DC_inds
    bcs = DirichletBC(V, 0, boundary_ids)
    uh = Function(V)
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    return(uh)


def Mark(msh, f, uh,V,tolerance):
     # based on the tutorial: https://www.firedrakeproject.org/demos/netgen_mesh.py.html
     W = FunctionSpace(msh, "DG", 0) # Both the error indicator and the marked element vector will be DG0 field.
     w = TestFunction(W)
     R_T = f+div(grad(uh))
     n = FacetNormal(V.mesh())
     h = CellDiameter(msh)
     R_dT = dot(grad(uh), n)
     # Assembling the error indicator eta
     eta = assemble(h**2*R_T**2*w*dx +0.5*(h("+")+h("-"))*(R_dT("+")+R_dT("-"))**2*(w("+")+w("-"))*dS)          
     # mark triangulation whose eta >= frac*eta_max
     frac = .96
     delfrac =0.02
     # keep marking triangulation when sum_marked eta< part *sum of eta
     part = .25
     mark = Function(W)
     # Filling in the marked element vector using eta.
     with mark.dat.vec as markedVec:
         with eta.dat.vec as etaVec:
             sum_eta = etaVec.sum()
             if sum_eta < tolerance:
                 return mark, sum_eta
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
     return mark, sum_eta

mesh=mesh0
it=0
sum_eta=1
while sum_eta>tolerance and it<max_iterations:
  x, y,z = SpatialCoordinate(mesh)
  f = Constant(-2.)
  V = FunctionSpace(mesh,"Lagrange",deg)
  uh = Esolver(mesh, f, V,DC_inds)
  mark,sum_eta = Mark(mesh, f,uh,V,tolerance)
  PETSc.Sys.Print("error indicator sum_eta is ", sum_eta)
  mesh = mesh.refine_marked_elements(mark)
  it=it+1
  PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize())
  
