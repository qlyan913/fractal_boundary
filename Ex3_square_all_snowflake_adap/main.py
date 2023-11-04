import os 
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
file=open('input.txt','w')
file.write(f"{n}\n")
file.write(f"{deg}")
file.close()

# run the script square_solver.py
#os.system('mpiexec -n 32 python3 square_solver.py')
os.system('python3 square_solver.py')
