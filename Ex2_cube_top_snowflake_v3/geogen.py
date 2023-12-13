# Qile Yan 2023-12-13
from netgen.occ import *
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
    # P1 ---- N1 ----- N2 ----- N3 ------P2
    # out put
    # array of shape 2x3. Four points N1, N2, N3, N4 on the line P1P2.

    P1 = vertices[0]
    P2 = vertices[1]
    dx = P2-P1
    N1 = P1+dx/4.
    N2 = N1+dx/4.
    N3 = N2+dx/4.
    return np.array([N1,N2,N3])

def divide_square(sq_vertices):
    # input
    # a sqaure  - array of shape 4 x 3. Each row is a vertices of the square which should be ordered as following graph.
    #
    # output
    # new_hole - the hole in the input square
    # new_main - the 5 sqaures (over the holde)  which will be outside boundary of a plane surface
    # new_other- the subsqaures of the input square other than the hole
    # order of saved each sub-squares in the array are as follows:
    #    P1 ------ N5----- N11 ----- N17 ----- P4 
    #    |         |        |         |         |         
    #    |         |        |         |         |         
    #    |         |        |         |         |         
    #    N1 ------ N6 ---- N12 ----  N18 ----- N24 
    #    |         |        |         |         |         
    #    |         |        |  hole   |         |         
    #    |         |        |         |         |         
    #    N2------- N7 ---- N13 ----- N19 ----- N25 
    #    |         |        |         |         |         
    #    |         |        |         |         |         
    #    |         |        |         |         |         
    #    N3------- N8 ---- N14 ----- N20 ----- N26 
    #    |         |        |         |         |         
    #    |         |        |         |         |        
    #    |         |        |         |         |          
    #    P2 ------ N9 ---- N15 ----  N21 ------ P3
   
    P1 = sq_vertices[0]
    P2 = sq_vertices[1]
    P3 = sq_vertices[2]
    P4 = sq_vertices[3]
    N1, N2,N3 =divide_line(np.array([P1,P2]))
    N9,N15,N21 =divide_line(np.array([P2,P3]))
    N26,N25,N24 =divide_line(np.array([P3,P4]))
    N17,N11,N5 =divide_line(np.array([P4,P1]))
    N6,N7,N8 =divide_line(np.array([N5,N9]))
    N12, N13, N14 =divide_line(np.array([N11,N15]))
    N18,N19,N20 = divide_line(np.array([N17,N21]))
    new_other=[np.array([P1,N1,N6,N5]),np.array([N1,N2,N7,N6]),np.array([N2,N3,N8,N7]),
                         np.array([N3,P2,N9,N8]),np.array([N5,N6,N12,N11]),
                         np.array([N6,N7,N13,N12]),np.array([N7,N8,N14,N13]),np.array([N8,N9,N15,N14]),np.array([N11,N12,N18,N17]),np.array([N13,N14,N20,N19]),np.array([N14,N15,N21,N20]),np.array([N17,N18,N24,P4]),np.array([N18,N19,N25,N24]),np.array([N19,N20,N26,N25]),np.array([N20,N21,P3,N26])
                         ]
    new_hole = [np.array([N12,N13,N19,N18])]
    normal = np.cross(P3-P2,P1-P2)
    normal = normal/np.linalg.norm(normal)
    ndx = np.linalg.norm(P2-P1)/4.
    H1 = N12+ndx*normal
    H2 = N13+ndx*normal
    H3 = N19+ndx*normal
    H4 = N18 +ndx*normal
    new_main=[np.array([H1,N12,N13,H2]),np.array([H2,N13,N19,H3]),np.array([H3,N19,N18,H4]),
              np.array([H4,N18,N12,H1]),np.array([H1,H2,H3,H4])]
    #                 H1---------H4
    #                /|         / |
    #             H2/---------H3  |
    #              |  N12-----｜--N18
    #              | /        ｜ /
    #              |/         ｜/
    #             N13---------N19
    small_cube=[N13,H4]
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

# define number of levels here
def MakeGeometry(fractal_level):
    cube =  Box(Pnt(0,0,0), Pnt(1,1,1))
    for i in range(6):
        if cube.faces[i].center[0]==0:
             cube.faces[i].name='left'
        elif cube.faces[i].center[0]==1:
             cube.faces[i].name='right'
        elif cube.faces[i].center[1]==0:
             cube.faces[i].name='front'
        elif cube.faces[i].center[1]==1:
             cube.faces[i].name='back'
        elif cube.faces[i].center[2] == 0:
             cube.faces[i].name = 'bot'
        else:
             cube.faces[i].name = 'top'
    fractal_domain=cube
    # Define the list of squares for the Koch snowflake    
    square0=np.array([[0,1,1],[0,0,1],[1,0,1],[1,1,1]])
    sq_list0=[[square0]]
    sq_cube_list  = koch_snowflake([sq_list0,[]],fractal_level)
    small_cube_list=sq_cube_list[1]

    for i in range(len(small_cube_list)):
       P1=small_cube_list[i][0]
       P2=small_cube_list[i][1]
       cube =  Box(Pnt(P1[0],P1[1],P1[2]), Pnt(P2[0],P2[1],P2[2]))
       cube.bc('top')
       fractal_domain= fractal_domain + cube

    geo = OCCGeometry(fractal_domain)
    return geo
