# -div ( D grad u) = 0 in Omega
# u = 1 on bottom
# du/dn= 0 on sides 
# Lambda du/dn+u=0 on top
#
#  Boundary surfaces are numbered as follows:
#  1: plane x == 0
#  2: plane x == 1
#  3: plane y == 0
#  4: plane y == 1
#  5: plane z == 0
#  6: plane z == 1 replaced by snowflake
#
# In this script, we calculate the flux through the top face, i.e., \int_top -D du/dn ds
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import csv
from netgen.occ import *
from geogen import *
from Ex2_solver import *
#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#deg=int(input("Enter the degree of polynomial: "))
nn=2
deg=2
tolerance = 1e-7
max_iterations = 4
# dimension of fractal boundary
dim_frac=np.log(13)/np.log(9)
l=(1/3.)**nn
Lp=(5./3.)**nn
# load the ngmesh
from netgen import meshing
ngmsh = meshing.Mesh(3) # create a 3-dimensional mesh
ngmsh.Load(f"domain/cube_{nn}.vol")


# Get label of boundary of Netgen mesh to index
bc_left = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['left']][0]
bc_right = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['right']][0]
bc_front = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['front']][0] 
bc_back = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['back']][0] 
bc_bot = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['bot']][0] 
bc_top = [i+1 for [i, name] in enumerate(ngmsh.GetRegionNames(codim=1)) if name in ['top']]


Phi=[] # the flux through the top face 
import numpy as np
def get_flux(ngmsh,LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top):
    cc=[]
    Phi=[]
    for Lambda in LL:
        ngmsh = meshing.Mesh(3) # create a 3-dimensional mesh
        ngmsh.Load(f"domain/cube_{nn}.vol")
        mesh=Mesh(ngmsh)
        it=0
        outfile = File(f"refined_mesh/fractal_mesh/ref_n{nn}_{it}.pvd")
        outfile.write(mesh)
        PETSc.Sys.Print(f"Refined mesh saved as refined_mesh/fractal_mesh/ref_n{nn}_{it}.pvd.")
        PETSc.Sys.Print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
        sum_eta=1
        while sum_eta>tolerance and it<max_iterations:
           x, y,z = SpatialCoordinate(mesh)
           D = Constant(1.)
           f = Constant(0.)
           u_D=Constant(1.)
           k1 =Constant(0.)
           k2 =Constant(0.)
           k3 =Constant(0.)
           k4 =Constant(0.)
           n = FacetNormal(mesh)
           l=  Constant(0.)
           V = FunctionSpace(mesh,"Lagrange",deg)
           PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize())
           uh = snowsolver(mesh, D, Lambda, f, u_D, k1, k2,k3,k4, l,V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
           mark,sum_eta = Mark_v2(mesh,Lambda, f,uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top)
           PETSc.Sys.Print("error indicator sum_eta is ", sum_eta)
           #PETSc.Sys.Print("Refining the marked elements ...")
           #Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(tuple(bc_top)))
           Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(tuple(bc_top)))
           mesh = mesh.refine_marked_elements(mark)
           it=it+1
           outfile = File(f"refined_mesh/fractal_mesh/ref_n{nn}_{it}.pvd")
           outfile.write(mesh)
           PETSc.Sys.Print(f"Refined mesh saved as refined_mesh/fractal_mesh/ref_n{nn}_{it}.pvd.")
           PETSc.Sys.Print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
           
        PETSc.Sys.Print("Lambda is ", Lambda, " flux is ", Phi_temp2)
        Phi.append(Phi_temp2)
 
    return Phi  

# get flux Phi0 at lambda = 0
Lambda=10**(-11)
Phi =get_flux(ngmsh,[Lambda],nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)

