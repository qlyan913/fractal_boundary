# Qile Yan 2024-01-29
# Solve
#   -\Delta u =f in Omega
# where Omega=Q\Q_int, Q=[0,1]x[0,1] with fractal boundary,
# Q_int =[0.45,0.55]x[0.45,0.55]
# u=g on \partial Q, u=g_int on \partial Q_int
# Adaptive FEM
from firedrake.petsc import PETSc
from firedrake import *
def snowsolver(mesh, f,g,g_int,V):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = (1,2,3,4) # 1:top 2:right 3:bottom 4:left
    # list of boundary ids that corresponds to the interior boundary of the domain
    boundary_ids_int = (5,6,7,8) # 5:top 6:right 7:bottom 8:left
    bc = DirichletBC(V, g, boundary_ids)
    bc_int = DirichletBC(V, g_int, boundary_ids_int)
    uh = Function(V,name="uh")
    #solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=[bc,bc_int], solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    #solve(a == L, uh, bcs=bcs)
    return(uh)

def get_solution(mesh,tolerance,max_iterations,deg):
    it=0
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "Lagrange", deg)
    f=Constant(0.0)
    g=0.0
    g_int=10.0
    uh = snowsolver(mesh, f,g,g_int,V)
    mark, sum_eta,eta_max = Mark(mesh,f,uh,V,tolerance)
    while sum_eta>tolerance and it<max_iterations:
        it=it+1
        mesh = mesh.refine_marked_elements(mark)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "Lagrange", deg)
        f=Constant(0.0)
        g=0.0
        uh = snowsolver(mesh, f,g,g_int,V)
        mark, sum_eta,eta_max = Mark(mesh,f,uh,V,tolerance)
        PETSc.Sys.Print("Refined Mesh with degree of freedom " , V.dof_dset.layout_vec.getSize(), 'sum_eta is ', sum_eta)
    return uh,mesh


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
