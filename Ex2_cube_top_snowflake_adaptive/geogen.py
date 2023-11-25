# Qile Yan 2023-11-14
from netgen.csg import *
import numpy as np
#  Create a mesh a cube where the top is replaced by snowflake
#  Boundary surfaces will be labelled as follows:
#  left: plane x == 0
#  right: plane x == 1
#  front: plane y == 0
#  back: plane y == 1
#  bot: plane z == 0
#  top: plane z == 1

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
    small_cube=[N10,N11,N9,H2]
    return (new_hole, new_other, new_main,small_cube)

def koch_snowflake(sq_cube, level):
    # sq_list, each element consists of {main surface,{holes},{others}}
    # "other" are list subsquares which will be divided in next step whose holes will be added to the main_surface.
    # In gmsh script, plane surface = {main surface, holes} means take the main surface as outside boundary
    # and the holes as inside boudary.
    sq_list=sq_cube[0]
    small_cube_list=sq_cube[1]
    if level == 0:
        return sq_cube
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
            temp_hole, temp_other, temp_main,small_cube=divide_square(s_others[j])
            new_holes=new_holes+temp_hole
            new_others=new_others+temp_other
            new_main=new_main+temp_main
            small_cube_list.append(small_cube)
       
        sq=[s_main,new_holes,new_others]
        new_sq_list.append(sq)
        for j in range(len(new_main)):
            new_sq_list.append([new_main[j]])   
    sq_cube=[new_sq_list,small_cube_list]
    return koch_snowflake(sq_cube, level - 1)

def Makecube_on_top(P1,P2,P4,P5):
    #                 P7----------P8
    #                /|         / |
    #             P5/---------P6  |
    #              |  P4------｜--P3
    #              | /        ｜ /
    #              |/         ｜/
    #             P1-----------P2
    n1=P1-P2
    n1=n1/np.linalg.norm(n1)
    left = Plane(Pnt(P1[0],P1[1],P1[2]), Vec(n1[0],n1[1],n1[2]))
    n1=P2-P1
    n1=n1/np.linalg.norm(n1)
    right = Plane(Pnt(P2[0],P2[1],P2[2]), Vec(n1[0],n1[1],n1[2])) 
    n1=P1-P4
    n1=n1/np.linalg.norm(n1)
    front = Plane(Pnt(P1[0],P1[1],P1[2]), Vec(n1[0],n1[1],n1[2]))
    n1=P4-P1
    n1=n1/np.linalg.norm(n1)
    back = Plane(Pnt(P4[0],P4[1],P4[2]), Vec(n1[0],n1[1],n1[2]))
    n1=P1-P5
    n1=n1/np.linalg.norm(n1)
    bot = Plane(Pnt(P1[0],P1[1],P1[2]), Vec(n1[0],n1[1],n1[2]))
    n1=P5-P1
    n1=n1/np.linalg.norm(n1)
    top = Plane(Pnt(P5[0],P5[1],P5[2]), Vec(n1[0],n1[1],n1[2]))
    cube = left * right * front * back * bot * top
    cube.bc("top")
    return cube

# define number of levels here
def MakeGeometry(fractal_level):
    left = Plane(Pnt(0, 0, 0), Vec(-1, 0, 0)).bc("left")
    right = Plane(Pnt(1, 1, 1), Vec(1, 0, 0)).bc("right")
    front = Plane(Pnt(0, 0, 0), Vec(0, -1, 0)).bc("front")
    back = Plane(Pnt(1, 1, 1), Vec(0, 1, 0)).bc("back")
    bot = Plane(Pnt(0, 0, 0), Vec(0, 0, -1)).bc("bot")
    top = Plane(Pnt(1, 1, 1), Vec(0, 0, 1)).bc("top")
    cube = left * right * front * back * bot * top
    fractal_domain=cube
 
    # Define the list of squares for the Koch snowflake    
    square0=np.array([[0,1,1],[0,0,1],[1,0,1],[1,1,1]])
    sq_list0=[[square0]]
    sq_cube_list  = koch_snowflake([sq_list0,[]],fractal_level)
    small_cube_list=sq_cube_list[1]
    for i in range(len(small_cube_list)):
       P1=small_cube_list[i][0]
       P2=small_cube_list[i][1] 
       P4=small_cube_list[i][2] 
       P5=small_cube_list[i][3] 
       small_cube=Makecube_on_top(P1,P2,P4,P5)
       fractal_domain = fractal_domain+small_cube

    geo = CSGeometry()
    geo.Add(fractal_domain)
    return geo
