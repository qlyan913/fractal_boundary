from math import sin,cos,sqrt,pi
from netgen.geom2d import SplineGeometry
from firedrake import *
#  Create a mesh where each sides are replaced by snowflake
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

# define number of levels here
def MakeGeometry(fractal_level):
    geo = SplineGeometry()
    ### Add points
    # top snowflake
    line_list,line_list2=koch_snowflake([],[np.array([[0,1],[1,1]])], n)
    line_list=line_list+line_list2
    for i in range(len(line_list)):
        l=line_list[i]
        P1=l[0]
        P2=l[1]
        geo.Append (["line", P1, P2, bc = "top")
  
    # right side snowflake
    line_list,line_list2=koch_snowflake([],[np.array([[1,1],[1,0]])], n)
    line_list=line_list+line_list2
    for i in range(len(line_list)):
        l=line_list[i]
        P1=l[0]
        P2=l[1]
        geo.Append (["line", P1, P2, bc = "right")
 
    # bottom snowflake
    line_list,line_list2=koch_snowflake([],[np.array([[1,0],[0,0]])], n)
    line_list=line_list+line_list2
    for i in range(len(line_list)):
        l=line_list[i]
        P1=l[0]
        P2=l[1]
        geo.Append (["line", P1, P2, bc = "bottom")
        
    # left side snowflake
    line_list,line_list2=koch_snowflake([],[np.array([[0,0],[0,1]])], n)
    line_list=line_list+line_list2
    for i in range(len(line_list)):
        l=line_list[i]
        P1=l[0]
        P2=l[1]
        geo.Append (["line", P1, P2, bc = "left")
    return geo
