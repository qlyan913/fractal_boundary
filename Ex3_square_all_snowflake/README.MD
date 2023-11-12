In this example, we solve

$-\Delta u=f\ \\text{on}\ \Omega$

$\Omega$: the unit square with each edge have been replaced by a square prefractal.

The boundary conditions is $u=0$ on $\partial \Omega$. 

1. To generate the domain, run `python3 snow_square.py`. Then it will require the parameter n, the number of iterations for snowflake.

2. To get the solution, run `python3 main.py`. The plot of solution and force term is saved as f_n.png and solution_n.png. The solution is saved to a .h5 file under the folder solutions/

3. To get the evaluation of the solution over a random path, run `python3 estimate.py`.




