"""
Solve
   -\Delta u =0 in Omega
with u = g_int on Omega_int, u=1 on Omega_ext
#  Boundary index are numbered as follows:
#  1: exterior boundary: bottom y=-1
#  2: exterior boundary: right x=2
#  3: exterior boundary: top y=2
#  4: exterior boundary: left x=-1
#  5: interior boundary of square
"""

import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from netgen.geom2d import SplineGeometry
from geogen import *
from hm_solver import *
from firedrake.pyplot import tripcolor
from matplotlib.ticker import PercentFormatter

#nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
#deg=int(input("Enter the degree of polynomial: "))
nn=2
deg=1
mesh_size=0.2
# choose a triangulation
geo = MakeGeometry(nn)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 30
# stop refinement when sum_eta less than tolerance
tolerance=1*1e-3
df=[] # mesh size, degree of freedom
err=[] #  L2 error of solution
err2=[] # H1 error 
sum_eta=1
it=0
while sum_eta>tolerance and it<max_iterations:
    it=it+1
    mesh=mesh0
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh0, "Lagrange", deg)
    u = 2 + x**2 + 3*x*y
    f=Constant(-2.0)
    g_int=2 + x**2
    g_bot=2 + x**2 - 3*x
    g_right=2 + 4 + 6*y
    g_top=2 + x**2 + 6*x
    g_left=2 + 1 - 3*y
    uh = sq_solver(mesh, f,g_int,g_bot,g_right,g_top,g_left,V)
    mark, sum_eta,eta_max = Mark(mesh,f,uh,V,tolerance)
    mesh0 = mesh.refine_marked_elements(mark)
    PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize(), 'sum_eta is ', sum_eta)
    df.append(V.dof_dset.layout_vec.getSize())
    err_temp=sqrt(assemble(dot(uh - u, uh - u) * dx))
    err.append(err_temp)
    err2_temp=sqrt(assemble(dot(uh - u, uh - u) * dx)+assemble(dot(grad(uh) -grad(u), grad(uh) - grad(u)) * dx))
    err2.append(err2_temp)
    print("Error of solution in L2 norm is ", err_temp)
    print("Error of solution in H1 norm is ", err2_temp)
import math
NN=np.array([(df[0]/df[i])**(1.)*err[0] for i in range(0,len(err))])
NN2=np.array([(df[0]/df[i])**(1./2.)*err2[0] for i in range(0,len(err))])
plt.figure()
plt.loglog(df, err,marker='o')
plt.loglog(df, err2,marker='s')
plt.loglog(df, NN)
plt.loglog(df, NN2)
plt.legend(['$L^2$ error', '$H^1$ error', '$O(dof^{-1})$','$O(dof^{-1/2})$'])
plt.xlabel('degree of freedom')
plt.title('Test')
plt.savefig(f"test_figs/test_{nn}.png")
print(f"Error vs mesh size saved to figures/test_{nn}.png")
plt.close()
