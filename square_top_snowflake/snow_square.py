import math
import numpy as np
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
mesh_size=input("Enter the meshsize: ")
# Define the Koch snowflake vertices
def koch_snowflake(vertices, level):
    # input
    #   vertices - real array of shape n x 2 (list of vertices)
    #   level - non-negative integer (number of steps of Koch snowflake generation
    # return
    # array of generated vertices  (m x 2)
    if level == 0:
        return vertices
    Rot=np.array([[.5, -np.sqrt(3.)/2],[np.sqrt(3.)/2, .5]]) # rotation matrix
    new_vertices = np.array([-1,-1]) # initiate the array
    for i in range(len(vertices) - 1):
        xl = vertices[i]
        xr = vertices[i + 1]
        # Calculate the new points
        dx = xr - xl
        ndx = np.linalg.norm(dx)
        xlm = xl + dx/3.
        xm = xlm + np.matmul(Rot,dx/3.) # rotate the vector 60 degree
        xrm = xlm + dx/3.
        new_vertices = np.vstack([new_vertices, xl, xlm, xm, xrm])
    # delete first row
    new_vertices = np.delete(new_vertices,0,0)
    new_vertices = np.vstack([new_vertices, vertices[-1]])  # Add the last point
    return koch_snowflake(new_vertices, level - 1)

# Parameters
#n =  4  # Number of iterations for Koch snowflake
#mesh_size =0.02# Mesh size
mesh_size2=mesh_size
# Define the vertices for the Koch snowflake
vertices_koch = koch_snowflake(np.array([[0, 1], [1, 1]]), n)

# Generate the Gmsh script
gmsh_script = f"""
SetFactory("OpenCASCADE");

// Define the unit square
Point(1) ={{1, 1, 0, {mesh_size}}};
Point(2) = {{1, 0, 0, {mesh_size}}};
Point(3) = {{0, 0, 0, {mesh_size}}};
Point(4) = {{0, 1, 0, {mesh_size}}};
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
// Define the Koch snowflake
"""
# Add Koch snowflake vertices and lines to the script
gmsh_script += "".join(f"Point({i + 5})={{{vertex[0]},{vertex[1]},0,{mesh_size2}}};\n Line({i + 4}) = {{{i + 4}, {i + 5}}};\n" for i, vertex in enumerate(vertices_koch[1:-1]))
gmsh_script += f"Line({2+len(vertices_koch)}) = {{{2+len(vertices_koch)},{1}}};\n"

# Define a line loop for the Koch snowflake
gmsh_script += "Curve Loop(1) = {" + \
    ", ".join([f"{i}" for i in range(1, len(vertices_koch) + 3)]) + \
    "};\nPlane Surface(1) = {1};\n"

gmsh_script += f"""
Physical Curve("Right") = {{1}};
Physical Curve("Bottom") = {{2}};
Physical Curve("Left") = {{3}};\n"""
gmsh_script += """Physical Curve("Top")= {""" + \
    ", ".join([f"{i}" for i in range(4, 4+len(vertices_koch) - 1)]) + \
    "};\n"

gmsh_script += f"""
Physical Surface("sur")={{1}};\n"""
# Save the Gmsh script to a file
with open("unit_square_with_koch.geo", "w") as f:
    f.write(gmsh_script)

print("Gmsh script has been generated and saved as 'unit_square_with_koch.geo'.")

import os
os.system('gmsh -2 unit_square_with_koch.geo')
print("Mesh file saved as 'unit_square_with_koch.msh'.")

import firedrake as fd
import matplotlib.pyplot as plt
# Load the Gmsh mesh file
mesh_file = 'unit_square_with_koch.msh'
fd_mesh = fd.Mesh(mesh_file)

# Plot the mesh
plt.figure()
fd.triplot(fd_mesh)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Koch Snowflake Mesh')
plt.show()
plt.savefig("snow.png")
