# Qile Yan 2023-10-22
import math
import numpy as np
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
mesh_size=input("Enter the meshsize: ")
#  Create a mesh a cube where the top is replaced by snowflake
#  Boundary surfaces are numbered as follows:
#  1: Top    x == 1
#  2: Right  y == 1
#  3: Bottom x == 0
#  4: Left  y == 0

def divide_line(vertices):
    # input
    # vertices - array of shape 2x3.
    #          - two points P1 P2
    # out put
    # array of shape 2x3. Two points N1 and N2 on the line P1P2.
    #
    #          N3-------N4
    #          |        |
    #          |        |
    #  P1 ---- N1 ----- N2 -----P2
    Rot=np.array([[0, -1],[1, 0]]) # rotation matrix
    P1 = vertices[0]
    P2 = vertices[1]
    dx = P2-P1
    N1 = P1+dx/3.
    N2 = N1+dx/3.
    N3 = N1 + np.matmul(Rot,dx/3.) # rotate the vector 60 degree
    N4 = N3 + dx/3. 
    return np.array([N1,N2,N3,N4])

def koch_snowflake(line_segments,line_to_be_divide, level):
    # line_segments: each element is a line of two points [P1,P2] those will never be divided again
    #                P1--------->---------P2
    # line_to_be_divide: each element is a line of two points [P1,P2]  those will be divided in next step
    if level == 0:
        return line_segments,line_to_be_divide
    new_line_to_be_divide=[]
    for i in range(len(line_to_be_divide)):
        ld = line_to_be_divide[i]
        P1 = ld[0]
        P2 = ld[1]
        N1, N2, N3, N4 =divide_line(np.array([P1,P2]))
        new_line=[np.array([P1,N1]),np.array([N2,P2])]
        for j in range(len(new_line)):
            line_segments.append(new_line[j])  
        new_line_tb_div=[np.array([N1,N3]),np.array([N3,N4]),np.array([N4,N2])]
        for j in range(len(new_line_tb_div)):
            new_line_to_be_divide.append(new_line_tb_div[j])
            
    return koch_snowflake(line_segments,new_line_to_be_divide, level - 1)

gmsh_script = f"""
//SetFactory("OpenCASCADE");
"""
pp=0
### Add points
# top snowflake
line_list,line_list2=koch_snowflake([],[np.array([[0,1],[1,1]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Point({pp+1})={{"
    gmsh_script += f"{P1[0]},{P1[1]},0,{mesh_size}}};\n"
    gmsh_script += f"Point({pp+2})={{"
    gmsh_script += f"{P2[0]},{P2[1]},0,{mesh_size}}};\n"
    pp=pp+2
  
# right side snowflake
line_list,line_list2=koch_snowflake([],[np.array([[1,1],[1,0]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Point({pp+1})={{"
    gmsh_script += f"{P1[0]},{P1[1]},0,{mesh_size}}};\n"
    gmsh_script += f"Point({pp+2})={{"
    gmsh_script += f"{P2[0]},{P2[1]},0,{mesh_size}}};\n"
    pp=pp+2

# bottom snowflake
line_list,line_list2=koch_snowflake([],[np.array([[1,0],[0,0]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Point({pp+1})={{"
    gmsh_script += f"{P1[0]},{P1[1]},0,{mesh_size}}};\n"
    gmsh_script += f"Point({pp+2})={{"
    gmsh_script += f"{P2[0]},{P2[1]},0,{mesh_size}}};\n"
    pp=pp+2
 
# left side snowflake
line_list,line_list2=koch_snowflake([],[np.array([[0,0],[0,1]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Point({pp+1})={{"
    gmsh_script += f"{P1[0]},{P1[1]},0,{mesh_size}}};\n"
    gmsh_script += f"Point({pp+2})={{"
    gmsh_script += f"{P2[0]},{P2[1]},0,{mesh_size}}};\n"
    pp=pp+2

pp=0
ll=0
##### add line
# top snowflake
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Line({ll+1}) ={{{pp+1},{pp+2}}};\n"
    pp=pp+2
    ll=ll+1
L1=int(ll)+1
top_list=range(1,L1)
# right side snowflake
line_list,line_list2=koch_snowflake([],[np.array([[1,1],[1,0]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Line({ll+1}) ={{{pp+1},{pp+2}}};\n"
    pp=pp+2
    ll=ll+1
L2=int(ll)+1
right_list=range(L1,L2)
# bottom snowflake
line_list,line_list2=koch_snowflake([],[np.array([[1,0],[0,0]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Line({ll+1}) ={{{pp+1},{pp+2}}};\n"
    pp=pp+2
    ll=ll+1
L3=int(ll)+1
bot_list=range(L2,L3)
# left side snowflake
line_list,line_list2=koch_snowflake([],[np.array([[0,0],[0,1]])], n)
line_list=line_list+line_list2
for i in range(len(line_list)):
    l=line_list[i]
    P1=l[0]
    P2=l[1]
    gmsh_script += f"Line({ll+1}) ={{{pp+1},{pp+2}}};\n"
    pp=pp+2
    ll=ll+1
L4=int(ll)+1
left_list=range(L3,L4)

gmsh_script+="""
Coherence;
"""
gmsh_script += "Curve Loop(1) = {" + \
    ", ".join([f"{i}" for i in range(1, ll+1)]) + \
    "};\nPlane Surface(1) = {1};\n"

gmsh_script += """Physical Curve("Top")= {""" + \
    ", ".join([f"{i}" for i in top_list]) + \
    "};\n"
gmsh_script += """Physical Curve("Right")= {""" + \
    ", ".join([f"{i}" for i in right_list]) + \
    "};\n"
gmsh_script += """Physical Curve("Bottom")= {""" + \
    ", ".join([f"{i}" for i in bot_list]) + \
    "};\n"
gmsh_script += """Physical Curve("Left")= {""" + \
    ", ".join([f"{i}" for i in left_list]) + \
    "};\n"

gmsh_script+="""
Physical Surface(1)={1};\n
Coherence;
"""


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

