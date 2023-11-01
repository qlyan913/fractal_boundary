from firedrake import *

for i in range(0, 4):
  with CheckpointFile(f"refined_cube_{i}.h5","r") as afile:
     mesh=afile.load_mesh()
  print("--refined mesh %d --" % (i))
  print("with %d elements and %d vertices " %(mesh.num_cells(), mesh.num_vertices()))
  print("max mesh size:", mesh.cell_sizes.dat.data.max())
  V = FunctionSpace(mesh,"Lagrange",1)
  print("degree of freedom of linear element:", V.dof_dset.layout_vec.getSize())
  V = FunctionSpace(mesh,"Lagrange",2)
  print("degree of freedom of quadratic element:", V.dof_dset.layout_vec.getSize())
