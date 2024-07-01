# Qile Yan 2024-06-07
# Solve -\Delta u =0 in Omega
# with u = g_int on Omega_int, u=g_ext on Omega_ext
# Adaptive FEM
from firedrake.petsc import PETSc
from firedrake import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from geogen import *
from scipy import stats
def sq_solver(mesh, f,g_int,g_bot,g_right,g_top,g_left,V):
#  The domain is \Omega=Q\Q_n where Q=[-1,1] x [-1, 1] and Q_n is the nth iteration of 1D cantor set inside Q.
#  Boundary index are numbered as follows:
#  1: exterior boundary: bottom y=-1
#  2: exterior boundary: right x=2
#  3: exterior boundary: top y=2
#  4: exterior boundary: left x=-1
#  5: interior boundary of square
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    bd_bot = (1)
    bc_bot = DirichletBC(V, g_bot,bd_bot)
    bd_right = (2)
    bc_right = DirichletBC(V, g_right,bd_right)
    bd_top = (3)
    bc_top = DirichletBC(V, g_top,bd_top)
    bd_left = (4)
    bc_left = DirichletBC(V, g_left,bd_left)
    bd_int = (5) # 2: interior
    bc_int = DirichletBC(V, g_int,bd_int)
    uh = Function(V,name="uh")
    #solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=[bc_bot,bc_right,bc_top,bc_left,bc_int], solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    #solve(a == L, uh, bcs=bcs)
    return(uh)


def Mark(msh, f, uh,V,tolerance):
     W = FunctionSpace(msh, "DG", 0)
     # Both the error indicator and the marked element vector will be DG0 field.
     w = TestFunction(W)
     R_T = f+div(grad(uh))
     n = FacetNormal(V.mesh())
     h = CellDiameter(msh)
     R_dT = dot(grad(uh), n)
     # Assembling the error indicator eta
     eta = assemble(h**2*R_T**2*w*dx +0.5*(h("+")+h("-"))*(R_dT("+")+R_dT("-"))**2*(w("+")+w("-"))*dS)
     # mark triangulation whose eta >= frac*eta_max
     frac = .9
     delfrac =0.02
     # keep marking triangulation when sum_marked eta< part *sum of eta
     part = .5
     mark = Function(W)
     # Filling in the marked element vector using eta.
     with mark.dat.vec as markedVec:
         with eta.dat.vec as etaVec:
             sum_eta = etaVec.sum()
             eta_max = etaVec.max()[1]
             if sum_eta < tolerance:
                 return mark, sum_eta,eta_max
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
     return mark, sum_eta, eta_max



def get_solution(mesh0,tolerance,max_iterations,deg):
   sum_eta=1
   it=0
   while sum_eta>tolerance and it<max_iterations:
      it=it+1
      mesh=mesh0
      x, y = SpatialCoordinate(mesh)
      V = FunctionSpace(mesh0, "Lagrange", deg)
      f=Constant(0.0)
      g_int=Constant(0.0)
      g_bot=Constant(1.0)
      g_right=Constant(1.0)
      g_top=Constant(1.0)
      g_left=Constant(1.0)
      uh = sq_solver(mesh, f,g_int,g_bot,g_right,g_top,g_left,V)
      mark, sum_eta,eta_max = Mark(mesh,f,uh,V,tolerance)
      mesh0 = mesh.refine_marked_elements(mark)
  # meshplot = triplot(mesh)
  # meshplot[0].set_linewidth(0.1)
  # meshplot[1].set_linewidth(1)
  # plt.xlim(-1, 2)
  # plt.axis('equal')
  # plt.title('Koch Snowflake Mesh')
  # plt.savefig(f"figures/snow_{n}_ref_{it}.pdf")
  # plt.close()
  # PETSc.Sys.Print(f"refined mesh plot saved to 'figures/snow_{n}_ref_{it}.pdf'.")
      PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize(), 'sum_eta is ', sum_eta)     
   return uh,f,V


def plot_colourline(x,y,c,indx_list):
   # col = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    #col = cm.jet(c)
    ax = plt.gca()
   # for idx in indx_list:
   #     for i in np.arange(len(idx)-1):
   #        ax.plot([x[idx[i]],x[idx[i+1]]], [y[idx[i]],y[idx[i+1]]], c=col[idx[i]],linewidth=0.3)
    im = ax.scatter(x, y, c=c, s=0.5,cmap='jet',vmin=0.95, vmax=1.05)
    return im

def plot_colourline_std(x,y,c,indx_list):
    col = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
   # col = cm.jet(c)
    ax = plt.gca()
    for idx in indx_list:
        for i in np.arange(len(idx)-1):
           ax.plot([x[idx[i]],x[idx[i+1]]], [y[idx[i]],y[idx[i+1]]], c=col[idx[i]],linewidth=0.3)
    im = ax.scatter(x, y, c=c, s=0,cmap=cm.jet)
    return im

def divide_line_N(vertices,N,n):
    # Input
    # vertices - array of shape 2x3.
    #          - two points P1 P2
    # N        - number of segments on the smallest edge (size of (1/3)^n/N)
    # Output
    # x_list -  array of shape Npx3. Np points at the center of segments over the line P1P2
    # nv     -  array of shape 1x3. the normal vector towards outside
    x_list=[]
    P1 = vertices[0][0]
    P2 = vertices[1][0]
    L_p1p2=np.linalg.norm(P1-P2)
    h=(1/3.)**n/N
    Np=int(L_p1p2/h)
    dx = (P2-P1)/Np
    x0=P1+dx/2.
    x_list.append(x0)
    for i in range(1,Np):
        x0=x0+dx
        x_list.append(x0)
    Rot=np.array([[0, -1],[1, 0]]) # rotation matrix
    nv=np.matmul(Rot,dx/3.)
    nv=nv/ np.linalg.norm(nv)
    return x_list, nv

def get_uu(uh,line_list,dy_list,N,l,n):
   xl_list=[]
   x_list=[]
   uu_all_list=[]
   # divide the bottom edge into segments with size (1/3)^n/N
   for L in line_list:
      pts_list,nv=divide_line_N(L,N,n)
      for pt in pts_list:
         x_list.append([pt,nv])
   alpha_list=[]
   for x in x_list:
      pt=x[0]
      nv=x[1]
      xl_list.append(pt-nv*l/2.)
      # sequence of points
      pp=[pt-yy*nv for yy in dy_list]
      uu=np.array(uh.at(pp))
      uu_all_list.append(uu)
   return  uu_all_list, xl_list,x_list

def get_alpha(uu_all,x_list,xl_list,dy_list):
   dy_list_log=np.log(dy_list)
   # estiamte alpha and c
   alpha_list=[]
   c_list=[]
   pt_xlist=[]
   pt_ylist=[]
   std_list=[]
   uu_list=[]
   xll_list=[]
   for i in range(len(uu_all)):
         uu=uu_all[i]
      #if min(uu)>0:
         uu_log=np.log(uu)
         alpha,b,std=std_linreg(dy_list_log, uu_log) 
         c=exp(b)
        # res = stats.linregress(dy_list_log,uu_log)
        # c=exp(res.intercept)*exp(-25)
        # alpha=res.slope
        # std=res.stderr
         alpha_list.append(alpha)
         c_list.append(c)
         pt=x_list[i][0]
         pt_xlist.append(pt[0])
         pt_ylist.append(pt[1])
         std_list.append(std)
         uu_list.append(uu)
         xll_list.append(xl_list[i])
   return uu_list,xll_list, alpha_list, c_list,std_list,pt_xlist,pt_ylist

def plot_regression(filename,uu,c,dy_list,alpha,uxl,l_half):
   tt=c*(dy_list)**alpha
   plt.figure()
   plt.loglog(dy_list,uu,'b.')
   plt.loglog(dy_list,tt)
   plt.loglog(l_half,uxl,'r*')
   plt.ylabel('evaluation of solution')
   plt.xlabel('distance to boundary')
   plt.legend(['value of solution','${%s}(dx)^{{%s}}$' % (c,alpha),'$u(x_l)$'])
   plt.savefig(filename)
   plt.close()
   print(f"plot of one example of solution in loglog is saved in ", filename)


def std_linreg(x,y):
    x_mean=np.mean(x)
    x_std=np.std(x)
    x_tilde=(x-x_mean)/x_std
    res = stats.linregress(x_tilde, y)
    std=res.stderr
    b_tilde=res.intercept
    a_tilde=res.slope
    a=a_tilde/x_std
    b=-a_tilde/x_std*x_mean+b_tilde
    return a,b, std

def std_linreg_v2(x,y):
    y_mean=np.mean(y)
    y_std=np.std(y)
    y_tilde=(y-y_mean)/y_std
    res = stats.linregress(x, y_tilde)
    std=res.stderr
    b_tilde=res.intercept
    a_tilde=res.slope
    a=a_tilde*y_std
    b=y_std*b_tilde+y_mean
    return a,b, std

def std_linreg_v3(x,y):
    y_mean=np.min(y)
    y_std=np.max(y)-np.min(y)
    y_tilde=(y-y_mean)/y_std
    res = stats.linregress(x, y_tilde)
    std=res.stderr
    b_tilde=res.intercept
    a_tilde=res.slope
    a=a_tilde*y_std
    b=y_std*b_tilde+y_mean
    return a,b, std
