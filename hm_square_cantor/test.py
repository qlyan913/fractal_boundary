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

nn=int(input("Enter the number of iterations for the pre-fractal boundary: "))
deg=int(input("Enter the degree of polynomial: "))
mesh_size=0.2
# choose a triangulation
geo = MakeGeometry(n)
ngmsh = geo.GenerateMesh(maxh=mesh_size)
mesh0 = Mesh(ngmsh)
# max of refinement
max_iterations = 40
# stop refinement when sum_eta less than tolerance
tolerance=1*1e-3
df=[] # mesh size, degree of freedom
err=[] # error of solution
sum_eta=1
it=0
while sum_eta>tolerance and it<max_iterations:
    it=it+1
    mesh=mesh0
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh0, "Lagrange", deg)
    u = 2 + x**2 + 3*x*y
    f=Constant(0.0)
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
import math
NN=np.array([1.1*(df[i]/df[0])**2*err[0] for i in range(0,len(err))])
plt.figure(1)
plt.loglog(df, err)
plt.loglog(df, NN)
plt.legend(['error', '$O(h^2)$'])
plt.xlabel('maximum of mesh size')
plt.title('Test')
plt.savefig("test_figs/test_3.png")
print("Error vs mesh size saved to figures/test_3.png")
plt.close()
