from firedrake import *
n=int(input("Enter the number of iterations for the pre-fractal boundary: "))
mesh_file =f'unit_cube_with_koch_n{n}.msh'
mesh=Mesh(mesh_file)
MH = MeshHierarchy(mesh, 3)
for i in range(0, len(MH)):
   mesh=MH[i]
   with CheckpointFile(f"refined_cube_{i}.h5",'w') as afile:
     afile.save_mesh(mesh)


