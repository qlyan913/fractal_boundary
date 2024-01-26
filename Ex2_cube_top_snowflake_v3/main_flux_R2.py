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
nn=3
deg=5
tolerance = 1e-4
max_iterations = 2
# dimension of fractal boundary
dim_frac=np.log(20)/np.log(4)
l=(1/4.)**nn
Lp=(6./4.)**nn
# load the ngmesh
from netgen import meshing
ngmsh = meshing.Mesh(3) # create a 3-dimensional mesh
ngmsh.Load(f"domain/cube_{nn}.vol")
#mesh_size=0.5
#geo = MakeGeometry(nn)
#ngmsh = geo.GenerateMesh(maxh=mesh_size)

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
#        ngmsh = geo.GenerateMesh(maxh=mesh_size)
        mesh=Mesh(ngmsh)
        it=1
        x, y,z = SpatialCoordinate(mesh)
        D = Constant(1.)
        f = Constant(0.)
        u_D=Constant(1.)
        V = FunctionSpace(mesh,"Lagrange",deg)
        PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize())
        #PETSc.Sys.Print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
        uh = snowsolver_v2(mesh, D, Lambda, f, u_D,V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
        mark,sum_eta = Mark(mesh, f,uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top)
        #mark,sum_eta =Mark_v2(mesh,Lambda, f, uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top)
        PETSc.Sys.Print("error indicator sum_eta is ", sum_eta)
        Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(tuple(bc_top)))
        while sum_eta>tolerance and it<max_iterations:
           mesh = mesh.refine_marked_elements(mark)
           x, y,z = SpatialCoordinate(mesh)
           D = Constant(1.)
           f = Constant(0.)
           u_D=Constant(1.)
         #  n = FacetNormal(mesh) 
           V = FunctionSpace(mesh,"Lagrange",deg)
           #PETSc.Sys.Print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
           PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize())
           #PETSc.Sys.Print(f'Finite element mesh has {mesh.num_cells()} cells and {mesh.num_vertices()} vertices.')
           uh = snowsolver_v2(mesh, D, Lambda, f, u_D,V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
           mark,sum_eta = Mark(mesh, f,uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top)
#           mark,sum_eta =Mark_v2(mesh,Lambda, f, uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top)
           PETSc.Sys.Print("error indicator sum_eta is ", sum_eta)
           #PETSc.Sys.Print("Refining the marked elements ...")
           #Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(tuple(bc_top)))
           Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(tuple(bc_top)))
           it=it+1
        PETSc.Sys.Print("Lambda is ", Lambda, " flux is ", Phi_temp2)
        Phi.append(Phi_temp2)

    return Phi  

# get flux Phi0 at lambda = 0
Lambda=10**(-11)
Phi =get_flux(ngmsh,[Lambda],nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
Phi0=Phi[0]
PETSc.Sys.Print('Phi0 is ', Phi0)
# Region 2: l<Lambda <L_p
l_log=np.log(l)
Lp_log=np.log(Lp)
LL_log = np.linspace(l_log,Lp_log,10) 
LL=np.exp(LL_log)
Phi =get_flux(ngmsh,LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top)
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1/dim_frac
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='red',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.xticks([10**(-2),10**(-1),1,l, Lp], ['$10^{-2}$','$10^{-1}$','$10^{0}$','l','Lp'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1/dim_frac})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R2.png")
PETSc.Sys.Print(f"Plot for l<Lambda<L_p saved to figures/Phi_Lam_{nn}_R2.png ")
with open(f'results/Phi_Lam_{nn}_R2.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i], 'flux': Phi[i] })
PETSc.Sys.Print(f"Result for l<Lambda<L_p saved to results/Phi_Lam_{nn}_R2.csv ")


