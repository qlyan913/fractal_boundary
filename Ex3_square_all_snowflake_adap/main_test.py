import os 
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
mesh_size=float(input("Enter the mesh size of the inital mesh:"))
deg=int(input("Enter the degree of polynomial in FEM space:"))
file=open('input_test.txt','w')
file.write(f"{n}\n")
file.write(f"{mesh_size}\n")
file.write(f"{deg}")
file.close()

# run the script 
os.system('mpiexec -n 32 python3 test-dirichlet-solver.py')

