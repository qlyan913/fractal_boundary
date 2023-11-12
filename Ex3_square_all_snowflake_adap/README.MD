In this example, we solve

$-\Delta u=f\ \\text{on}\ \Omega$

$\Omega$: the unit square with each edge have been replaced by a square prefractal.

The boundary conditions is $u=0$ on $\partial \Omega$. 

The adaptive code is based on the firedrake tutorial here: https://www.firedrakeproject.org/demos/netgen_mesh.py.html. 

1. In test_Lshape.py, it test the adaptive solver on the L shape domain. 
2. In test.py, it solving the PDE on square with fractal boundary. 






