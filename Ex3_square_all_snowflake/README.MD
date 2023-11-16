In this example, we solve

$-\Delta u=f\ \\text{on}\ \Omega$

$\Omega$: the unit square with each edge have been replaced by a square prefractal.

The boundary conditions is $u=0$ on $\partial \Omega$. 

1. First, we need to generate the domain by run `python3 snow_square.py`. It will require the parameter n, the number of iterations for snowflake.

2. To get the solution, run `python3 main.py`. The plot of solution and force term is saved to  figures/f_n.png and figures/solution_n.png. The solution is saved to a .h5 file under the folder solutions/

3. To get the evaluation of the solution over a random path, run `python3 estimate.py`.




