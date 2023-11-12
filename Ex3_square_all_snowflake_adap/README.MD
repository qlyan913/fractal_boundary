In this example, we solve

$-\Delta u=f\ \\text{on}\ \Omega$

$\Omega$: the unit square with each edge have been replaced by a square prefractal.

The boundary conditions is $u=0$ on $\partial \Omega$. 

Here, we implement with the adpative FEM. The adaptive meshing code is based on the firedrake tutorial here: https://www.firedrakeproject.org/demos/netgen_mesh.py.html. 

1. In test_Lshape.py, it test the adaptive solver on the L shape domain. The manufacuted solution is $u(r,\theta)=r^{2/3}sin(2/3\theta)$. The error v.s. degree of freedom is saved in figures/Lshape_test_dof.png. The refined mesh is saved in refined_mesh/Lshape_mesh.

2. In test.py, it solving the PDE on square with fractal boundary. 






