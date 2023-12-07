In this example, we solve

$-\text{div}(D\text{grad}u)=0\ \\text{on}\ \Omega$

$\Omega$: the unit square wherethe top edge has been replaced by a triangle prefractal.

The boundary conditions

On bottom edge, Dirichlet: $u=1$.

On sides, homogeneous Neumann: $\frac{\partial u}{\partial n}=0$.

On prefractal top edge, Robin boundary conditions: $\Lambda\frac{\partial u}{\partial n}+u=0$.

1. To generate the domain, run `python3 snow_square.py`. Then it will require the number of iterations for snowflake.

2. In the script main_flux.py, it solves the PDE and draws the plot for the flux through the top edge as a function of $\Lambda$. 


