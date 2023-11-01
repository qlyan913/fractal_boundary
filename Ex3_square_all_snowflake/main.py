import os 
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
file=open('input.txt','w')
file.write(f"{n}\n")
file.write(f"{deg}")
file.close()

os.system('mpiexec -n 32 python3 h_solver.py')
