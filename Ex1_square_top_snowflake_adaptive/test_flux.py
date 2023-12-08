from firedrake.petsc import PETSc
import csv
import matplotlib.pyplot as plt
from firedrake import *
import statistics 
from scipy import stats
from netgen.geom2d import SplineGeometry
from geogen import *
from Ex1_solver import *
mesh_size=1
deg=3
# solve PDE on unit square
# solution is u=1-1/(1+Lambda)y
# flux through top is Phi=-1/(1+Lambda)
tolerance = 1e-7
max_iterations = 10
bc_top=(1)
bc_right=(2)
bc_bot=(3)
bc_left=(4)
geo = SplineGeometry()
p1=[np.array([0,1]),0]
p2=[np.array([1,1]),1]
p3=[np.array([1,0]),2]
p4=[np.array([0,0]),3]
P1=geo.AppendPoint(*p1[0])
P2=geo.AppendPoint(*p2[0])
P3=geo.AppendPoint(*p3[0])
P4=geo.AppendPoint(*p4[0])    
# top y==1
geo.Append (["line", P2, P1], bc = 1)
# left x==0
geo.Append (["line", P1, P4], bc = 4)
# bot y==0
geo.Append (["line", P4, P3], bc = 3)
# right x==1
geo.Append (["line", P3, P2], bc = 2)
geo = MakeGeometry(nn)
def get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top):
    Phi=[]
    for Lambda in LL:
        ngmsh = geo.GenerateMesh(maxh=mesh_size)
        mesh = Mesh(ngmsh)
        it = 0
        sum_eta=1
        while sum_eta>tolerance and it<max_iterations:
            x, y = SpatialCoordinate(mesh)
            n = FacetNormal(mesh)
            D = Constant(1.)
            f = Constant(0.)
            u_D=Constant(1.)
            kl=Constant(0.)
            kr=Constant(0.)
            l=Constant(0.)
            V = FunctionSpace(mesh,"Lagrange",deg)
            uh = snowsolver(mesh, D, Lambda, f, u_D, kl, kr, l,deg,bc_right,bc_bot,bc_left,bc_top)
            mark, sum_eta = Mark(mesh, f,V,uh,tolerance)
#            mark,sum_eta=Mark_v2(mesh,Lambda, f, uh,V,tolerance,bc_left,bc_right,bc_top)
            PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize(), 'sum_eta is ', sum_eta)
            it=it+1
            Phi_temp=assemble(-Constant(D)*inner(grad(uh), n)*ds(bc_top))
            Phi_temp2=assemble(Constant(D)/Constant(Lambda)*uh*ds(bc_top))
            mesh = mesh.refine_marked_elements(mark)
        PETSc.Sys.Print("Lambda is ", Lambda, " flux is ", Phi_temp2)
        Phi.append(Phi_temp2)
    return Phi
# get flux Phi0 at lambda = 0
Lambda=10**(-11)
Phi=get_flux(geo, [Lambda],nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
Phi0=Phi[0]
PETSc.Sys.Print("phi0 is", Phi0)

# calculate flux for various Lambda 
Phi=[]
import numpy as np
LL = np.array([2**i for i in range(-15,10)])
Phi=get_flux(geo, LL,nn,deg,tolerance,max_iterations,bc_left,bc_right,bc_bot,bc_top)
with open(f'results/Phi_Lam_{nn}.csv', 'w', newline='') as csvfile:
    fieldnames = ['Lambda', 'flux', 'flux_u']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'Lambda': 0, 'flux': Phi0_exact, 'flux_u':Phi0})
    for i in range(len(LL)):
       writer.writerow({'Lambda': LL[i],'flux':Phi_exact[i] , 'flux_u': Phi[i]})
   
PETSc.Sys.Print(f"Result for 0<Lambda<1000 saved to results/Phi_Lam_{nn}.csv ")
