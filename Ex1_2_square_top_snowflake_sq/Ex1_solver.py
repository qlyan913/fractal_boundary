# Qile Yan 2023-12-7
# -div ( D grad u) = f in Omega
# u = g on bottom
# du/dn = kl / kr  on left and right
# Lambda du/dn + u = l on top
#
# Adaptive FEM
from firedrake.petsc import PETSc
from firedrake import *
def snowsolver(mesh, D, Lambda, f, g, kl, kr, l,deg,bc_right,bc_bot,bc_left,bc_top):
    V = FunctionSpace(mesh, "Lagrange", deg)
    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    a = Constant(D)*dot(grad(u), grad(v))*dx+Constant(D)/Constant(Lambda)*u*v*ds(tuple(bc_top))
    L = f*v*dx + kl*v*ds(bc_left) + kr*v*ds(bc_right) + (1./Lambda)*l*v*ds(tuple(bc_top))
    boundary_ids = bc_bot
    bcs = DirichletBC(V, g, boundary_ids)
    uh = Function(V)
    solve(a == L, uh, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
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
                 return mark, sum_eta
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
