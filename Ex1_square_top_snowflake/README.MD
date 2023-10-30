In this example, we solve

$-\text{div}(D\text{grad}u)=0\ \\text{on}\ \Omega$

$\Omega$: the unit square wherethe top edge has been replaced by a triangle prefractal.

The boundary conditions

On bottom edge, Dirichlet: $u=1$.

On sides, homogeneous Neumann: $\frac{\partial u}{\partial n}=0$.

On prefractal top edge, Robin boundary conditions: $\Lambda\frac{\partial u}{\partial n}+u=0$.

1. To generate the domain, run python3 snow_square.py `git status` 


