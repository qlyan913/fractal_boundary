In this example, we solve

$-\Delta u=f\ \\text{on}\ \Omega$

$\Omega$: the unit square with each edge have been replaced by a square prefractal.

The boundary conditions is $u=0$ on $\partial \Omega$. 

The adaptive code is based on the firedrake tutorial here: https://www.firedrakeproject.org/demos/netgen_mesh.py.html. 

1. To generate the domain, run `python3 snow_square.py`. Then it will require the parameter n, the number of iterations for snowflake.

2. To get the solution, run `python3 main.py`. The plot of solution and force term is saved as f_n.png and solution_n.png

3. The evaluation of solution at a sequence points is saved in figures/evaluate_n.png and figures/evaluate_log_n.png




