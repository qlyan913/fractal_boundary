# Qile Yan 2023-09-29
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
mesh_size=input("Enter the meshsize: ")
import math
import numpy as np
#  Create a mesh a cube where the top is replaced by snowflake
#  Boundary surfaces are numbered as follows:
#  1: plane x == 0
#  2: plane x == 1
#  3: plane y == 0
#  4: plane y == 1
#  5: plane z == 0
#  6: plane z == 1
def divide_line(vertices):
    # input
    # vertices - array of shape 2x3.
    #          - two points P1 P2
    # P1 ---- N1 ----- N2 -----P2
    # out put
    # array of shape 2x3. Two points N1 and N2 on the line P1P2.

    P1 = vertices[0]
    P2 = vertices[1]
    dx = P2-P1
    N1 = P1+dx/3.
    N2 = N1+dx/3.
    return np.array([N1,N2])

def divide_square(sq_vertices):
    # input
    # a sqaure  - array of shape 4 x 3. Each row is a vertices of the square which should be ordered as following graph.
    #
    # output
    # new_hole - the hole in the input square  
    # new_main - the 5 sqaures (over the holde)  which will be outside boundary of a plane surface
    # new_other- the subsqaures of the input square other than the hole 
    # order of saved each sub-squares in the array are as follows:
    #    P1 ------ N8 ----- N7 ----  P4
    #    |         |        |         |
    #    |         |        |         |
    #    |         |        |         |
    #    N1 ------ N9 ----- N12 ----  N6
    #    |         |        |         |
    #    |         |  hole  |         |
    #    |         |        |         |
    #    N2 ------ N10 ---- N11 ----  N5
    #    |         |        |         |
    #    |         |        |         |
    #    |         |        |         |
    #    P2 ------ N3 ----- N4 -----  P3
    #
    P1 = sq_vertices[0]
    P2 = sq_vertices[1]
    P3 = sq_vertices[2]
    P4 = sq_vertices[3]
    N1, N2 =divide_line(np.array([P1,P2]))
    N3, N4 =divide_line(np.array([P2,P3]))
    N5, N6 =divide_line(np.array([P3,P4]))
    N7, N8 =divide_line(np.array([P4,P1]))
    N9, N10 =divide_line(np.array([N8,N3]))
    N11, N12 =divide_line(np.array([N4,N7]))
    new_other=[np.array([P1,N1,N9,N8]),np.array([N1,N2,N10,N9]),np.array([N2,P2,N3,N10]),
                         np.array([N10,N3,N4,N11]),np.array([N11,N4,P3,N5]),np.array([N12,N11,N5,N6]),
                         np.array([N7,N12,N6,P4]),np.array([N8,N9,N12,N7])
                         ]
    new_hole = [np.array([N9,N10,N11,N12])]
    normal = np.cross(P3-P2,P1-P2)
    normal = normal/np.linalg.norm(normal)
    ndx = np.linalg.norm(P2-P1)/3.
    H1 = N9+ndx*normal
    H2 = N10+ndx*normal
    H3 = N11+ndx*normal
    H4 = N12 +ndx*normal
    new_main=[np.array([H1,N9,N10,H2]),np.array([H2,N10,N11,H3]),np.array([H3,N11,N12,H4]),
              np.array([H4,N12,N9,H1]),np.array([H1,H2,H3,H4])]

    return (new_hole, new_other, new_main)

def koch_snowflake(sq_list, level):
    # sq_list, each element consists of {main surface,{holes},{others}}
    # "other" are list subsquares which will be divided in next step whose holes will be added to the main_surface.
    # In gmsh script, plane surface = {main surface, holes} means take the main surface as outside boundary
    # and the holes as inside boudary. 
    if level == 0:
        return sq_list
    new_sq_list=[]
    for i in range(len(sq_list)):
        sq = sq_list[i]
        s_main = sq[0]
        new_others = []
        new_main = []
        if len(sq)>1:
            s_holes = sq[1]
            s_others = sq[2]
            new_holes = s_holes
        else:
            new_holes = []
            s_others = [s_main]

        for j in range(len(s_others)):
            temp_hole, temp_other, temp_main=divide_square(s_others[j])
            new_holes=new_holes+temp_hole
            new_others=new_others+temp_other
            new_main=new_main+temp_main

        sq=[s_main,new_holes,new_others]
        new_sq_list.append(sq)
        for j in range(len(new_main)):
            new_sq_list.append([new_main[j]])

    return koch_snowflake(new_sq_list, level - 1)

# Parameters
#n =  3  # Number of iterations for Koch snowflake
#mesh_size = 0.1  # Mesh size
mesh_size2= mesh_size
# Define the list of squares for the Koch snowflake
square0=np.array([[0,1,1],[0,0,1],[1,0,1],[1,1,1]])
sq_list0=[[square0]]
sq_list=koch_snowflake(sq_list0, n)

# Generate the Gmsh script
gmsh_script = f"""
//SetFactory("OpenCASCADE");
// Boundary surfaces are numbered as follows:
// 1: plane x == 0
// 2: plane x == 1
// 3: plane y == 0
// 4: plane y == 1
// 5: plane z == 0
// 6: plane z == 1
// Define the unit square
Point(1) ={{1, 1, 0, {mesh_size}}};
Point(2) = {{1, 0, 0, {mesh_size}}};
Point(3) = {{0, 0, 0, {mesh_size}}};
Point(4) = {{0, 1, 0, {mesh_size}}};
Point(5) = {{0, 1, 1, {mesh_size}}};
Point(6) = {{0, 0, 1, {mesh_size}}};
Point(7) = {{1, 0, 1, {mesh_size}}};
Point(8) = {{1, 1, 1, {mesh_size}}};
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};
Line(5) = {{5, 6}};
Line(6) = {{6,7}};
Line(7) = {{7,8}};
Line(8) = {{8,5}};
Line(9) = {{1,8}};
Line(10) = {{2,7}};
Line(11) = {{3,6}};
Line(12) = {{4,5}};
Curve Loop (1) = {{3,12,5,-11}};
Plane Surface (1) ={{1}};
Curve Loop (2) = {{1,10,7,-9}};
Plane Surface (2) ={{2}};
Curve Loop (3) = {{2,11,6,-10}};
Plane Surface (3) ={{3}};
Curve Loop (4) = {{4,9,8,-12}};
Plane Surface (4) ={{4}};
Curve Loop (5) = {{1,2,3,4}};
Plane Surface (5) ={{5}};
"""
pp=8
CL=6
PS=6
LL=12
for i in range (len(sq_list)):
    sq=sq_list[i]
    s_main = sq[0]
    pp0=pp
    LL0=LL
    CL0=CL
    gmsh_script += f"Point({pp+1})={{"
    gmsh_script += f"{s_main[0][0]},{s_main[0][1]},{s_main[0][2]},{mesh_size2}}};\n"
    gmsh_script += f"Point({pp+2})={{"
    gmsh_script += f"{s_main[1][0]},{s_main[1][1]},{s_main[1][2]},{mesh_size2}}};\n"
    gmsh_script += f"Point({pp+3})={{"
    gmsh_script += f"{s_main[2][0]},{s_main[2][1]},{s_main[2][2]},{mesh_size2}}};\n"
    gmsh_script += f"Point({pp+4})={{"
    gmsh_script += f"{s_main[3][0]},{s_main[3][1]},{s_main[3][2]},{mesh_size2}}};\n"
    gmsh_script += f"Line({LL+1})={{{pp+1},{pp+2}}};\n"
    gmsh_script += f"Line({LL+2})={{{pp+2},{pp+3}}};\n"
    gmsh_script += f"Line({LL+3})={{{pp+3},{pp+4}}};\n"
    gmsh_script += f"Line({LL+4})={{{pp+4},{pp+1}}};\n"
    gmsh_script += f"Curve Loop({CL})={{{LL+1},{LL+2},{LL+3},{LL+4}}};\n"
    pp = pp+4
    LL=LL+4
    CL=CL+1
    if len(sq)>1:
        holes = sq[1]
        for j in range(len(holes)):
            gmsh_script += f"Point({pp+1})={{"
            gmsh_script += f"{holes[j][0][0]},{holes[j][0][1]},{holes[j][0][2]},{mesh_size2}}};\n"
            gmsh_script += f"Point({pp+2})={{"
            gmsh_script += f"{holes[j][1][0]},{holes[j][1][1]},{holes[j][1][2]},{mesh_size2}}};\n"
            gmsh_script += f"Point({pp+3})={{"
            gmsh_script += f"{holes[j][2][0]},{holes[j][2][1]},{holes[j][2][2]},{mesh_size2}}};\n"
            gmsh_script += f"Point({pp+4})={{"
            gmsh_script += f"{holes[j][3][0]},{holes[j][3][1]},{holes[j][3][2]},{mesh_size2}}};\n"
            gmsh_script += f"Line({LL+1})={{{pp+1},{pp+2}}};\n"
            gmsh_script += f"Line({LL+2})={{{pp+2},{pp+3}}};\n"
            gmsh_script += f"Line({LL+3})={{{pp+3},{pp+4}}};\n"
            gmsh_script += f"Line({LL+4})={{{pp+4},{pp+1}}};\n"
            gmsh_script += f"Curve Loop({CL})={{{LL+1},{LL+2},{LL+3},{LL+4}}};\n"
            CL=CL+1
            pp=pp+4
            LL=LL+4
        gmsh_script += f"Plane Surface({PS})={{"+ \
         ", ".join([f"{i}" for i in range(CL0, CL)]) + \
         "};\n"
        PS=PS+1
    else:
        gmsh_script += f"Plane Surface({PS})={{{CL-1}}};\n"
        PS=PS+1

gmsh_script += f"Surface Loop(1)={{"+ \
         ", ".join([f"{i}" for i in range(1, PS)]) + \
         "};\n"

gmsh_script +="""
Volume (1)={1};
Physical Surface (1)={1};
Physical Surface (2)={2};
Physical Surface (3)={3};
Physical Surface (4)={4};
Physical Surface (5)={5};
"""
gmsh_script += f"Physical Surface(6)={{"+ \
         ", ".join([f"{i}" for i in range(6, PS)]) + \
         "};\n"
gmsh_script+="""
Physical Volume (1)={1};
Coherence;
"""

# Save the Gmsh script to a file
with open(f"unit_cube_with_koch_n{n}.geo", "w") as f:
    f.write(gmsh_script)

print(f"Gmsh script has been generated and saved as 'unit_cube_with_koch_n{n}.geo'.")

import os
os.system(f'gmsh -3 unit_cube_with_koch_n{n}.geo')

import firedrake as fd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Load the Gmsh mesh file
mesh_file = f'unit_cube_with_koch_n{n}.msh'
fd_mesh = fd.Mesh(mesh_file)
print(f"Mesh file  saved as 'unit_cube_with_koch_n{n}.msh'.")
# Plot the mesh
fig=plt.figure()
ax = plt.axes(projection='3d')
fd.triplot(fd_mesh,axes=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Koch Snowflake Mesh on Cube')
plt.show()
plt.savefig(f"snow_cube_n{n}.pdf")

outfile = fd.File(f"cube_{n}.pvd")
outfile.write(fd_mesh)
print(f"File for visualization in Paraview saved as 'cube_{n}_0.vtu'.")
