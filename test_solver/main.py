import time
from firedrake import * 
from firedrake.petsc import PETSc
mesh = UnitSquareMesh(100, 100)
deg=5
V = FunctionSpace(mesh, "CG", deg)
PETSc.Sys.Print("deggre of freedom", V.dof_dset.layout_vec.getSize())
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx
u = Function(V)
start=time.process_time()
solve(a == L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})
end=time.process_time()
PETSc.Sys.Print("cg method")
PETSc.Sys.Print("time used:", end-start)

start=time.process_time()
solve(a == L,u, solver_parameters={'ksp_type': 'preonly','pc_type': 'lu'}) # solve direclty using LU factorization
end=time.process_time()
PETSc.Sys.Print("solve with LU factorization")
PETSc.Sys.Print("time used:", end-start)


start=time.process_time()
solve(a==L,u)
end=time.process_time()
PETSc.Sys.Print("default direct solver")
PETSc.Sys.Print("time used:", end-start)


start=time.process_time()
solve(a == L, u, solver_parameters={'pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
end=time.process_time()
PETSc.Sys.Print("solve with boomeramg preconditioner")
PETSc.Sys.Print("time used:", end-start)

start=time.process_time()
solve(a == L, u, solver_parameters={'ksp_type': 'cg','pc_type': 'hypre','pc_hypre_type': 'boomeramg'})
end=time.process_time()
PETSc.Sys.Print("cg method with boomeramg preconditioner")
PETSc.Sys.Print("time used:", end-start)

