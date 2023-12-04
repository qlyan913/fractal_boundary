import os 
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_size=float(input("Enter the meshsize for initial mesh: "))
deg=int(input("Enter the degree of polynomial in FEM space:"))
file=open('input.txt','w')
file.write(f"{n}\n")
file.write(f"{mesh_size}\n")
file.write(f"{deg}")
file.close()

# run the script square_solver.py
#os.system('mpiexec -n 32 python3 harmonic_problem.py ')
os.system('python3 harmonic_problem.py')
