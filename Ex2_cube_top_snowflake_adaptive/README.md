In this example, we solve

$-\text{div}(D\text{grad}u)=0\ \\text{on}\ \Omega$

$\Omega$: the unit cube where the top edge has been replaced by a triangle prefractal.

The boundary conditions

On bottom surface, Dirichlet: $u=1$.

On sides, homogeneous Neumann: $\frac{\partial u}{\partial n}=0$.

On prefractal top surface, Robin boundary conditions: $\Lambda\frac{\partial u}{\partial n}+u=0$.

1. To generate the domain, run `python3 generate_mesh.py`. Then it will require the number of iterations for snowflake.

2. Run `python3 test.py` to test the adaptive meshing code.

3. In the script main_flux.py, it solves the PDE and draws the plot for the flux through the top surface as a function of $\Lambda$. 
