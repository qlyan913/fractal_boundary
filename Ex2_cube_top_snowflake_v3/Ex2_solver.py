# Qile Yan 2023-11-13
# -div ( D grad u) = f in Omega
# u = g on bottom
# du/dn = ki,  ki is function defined on boundary 1<= i<= 4 
# Lambda du/dn + u = l on top
#
#  Weak formulation:
#
#  Find u in H1 with u = g on bottom such that
#
#    \int D grad(u).grad(v) dx  +  \int_{top} D/Lambda u v ds
#         = \int f v dx  +  \int_{top} (1/Lambda) l v ds
#           +  \int_{left} k1 v ds   +  \int_{right} k2 v ds
#           +  \int_{front} k3 v ds   +  \int_{back} k4 v ds           
#
#  for all v in H1 which vanish on bottom.
#
#  Boundary surfaces are numbered as follows:
#  bc_left: plane x == 0
#  bc_right: plane x == 1
#  bc_front: plane y == 0
#  bc_back: plane y == 1
#  bc_bot: plane z == 0
#  bc_top: plane z == 1 replaced by snowflake
#  Adaptive FEM
from firedrake.petsc import PETSc
from firedrake import *

def snowsolver(mesh, D, Lambda, f, g, k1, k2,k3,k4, l, V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(tuple(bc_top))
    L = f*v*dx  + k1*v*ds(bc_left)+k2*v*ds(bc_right)+k3*v*ds(bc_front) + k4*v*ds(bc_back) + (1./Lambda)*l*v*ds(tuple(bc_top))

    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = bc_bot
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V)
#    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    return(uh)

def snowsolver_v2(mesh, D, Lambda, f, g, V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(tuple(bc_top))
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = bc_bot
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V)
#    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
    return(uh)

def snowsolver_v3(mesh, D, f, g, V,bc_left,bc_right,bc_front,bc_back,bc_bot,bc_top):
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Constant(D)*dot(grad(u), grad(v))*dx
    L = f*v*dx
    # list of boundary ids that corresponds to the exterior boundary of the domain
    boundary_ids = bc_bot
    bcs = DirichletBC(V, g, boundary_ids)
    bcs2= DirichletBC(V, 0, bc_top)
    uh = Function(V)
#    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    solve(a == L, uh, bcs=[bcs,bcs2], solver_parameters={'ksp_type': 'cg', 'pc_type': 'hypre','pc_hypre_type': 'boomeramg',"ksp_rtol":1e-6})
    return(uh)
  
    
def Mark(msh, f, uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top):
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
     part = .35
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


def Mark_v2(msh,Lambda, f, uh,V,tolerance,bc_left,bc_right,bc_front,bc_back,bc_top):
     W = FunctionSpace(msh, "DG", 0)
     # Both the error indicator and the marked element vector will be DG0 field.
     w = TestFunction(W)
     R_T = f+div(grad(uh))
     n = FacetNormal(V.mesh())
     h = CellDiameter(msh)
     R_dT = dot(grad(uh), n)
     R_dT_top=Lambda*dot(grad(uh), n)+uh
     # Assembling the error indicator eta
     eta = assemble(h**2*R_T**2*w*dx +0.25*(h("+")+h("-"))*(R_dT("+")+R_dT("-"))**2*(w("+")+w("-"))*dS
           +h*(R_dT)**2*(w)*ds(bc_right)
           +h*(R_dT)**2*(w)*ds(bc_right)
           +h*(R_dT)**2*(w)*ds(bc_front)
           +h*(R_dT)**2*(w)*ds(bc_back)
           +h*(R_dT_top)**2*(w)*ds(tuple(bc_top)))
     # mark triangulation whose eta >= frac*eta_max
     frac = .98
     delfrac =0.02
     # keep marking triangulation when sum_marked eta< part *sum of eta
     part = .2
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

