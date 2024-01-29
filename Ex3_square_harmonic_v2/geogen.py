from netgen.geom2d import SplineGeometry
import numpy as np
#  Create a mesh where each sides are replaced by snowflake
#  Boundary surfaces are numbered as follows:
#  1: Top    x == 1
#  2: Right  y == 1
#  3: Bottom x == 0
#  4: Left  y == 0
def divide_line(vertices,num_pts):
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
    P1 = vertices[0][0]
    P2 = vertices[1][0]
    dx = P2-P1
    N1 = P1+dx/3.
    N2 = N1+dx/3.
    N3 = N1 + np.matmul(Rot,dx/3.) # rotate the vector 60 degree
    N4 = N3 + dx/3.
    New_p1=[N1,num_pts+1]
    New_p2=[N2,num_pts+2]
    New_p3=[N3,num_pts+3]
    New_p4=[N4,num_pts+4]
    return New_p1,New_p2,New_p3,New_p4

def koch_snowflake(new_pts,num_pts,line_segments,line_to_be_divide, level):
    # pts:           list of points, each element consists of position of the point and the index of the point. 
    # line_segments: each element is a line of two points [P1,P2] those will never be divided again
    #                P1--------->---------P2
    # line_to_be_divide: each element is a line of two points [P1,P2]  those will be divided in next step
    if level == 0:
        return new_pts,num_pts,line_segments,line_to_be_divide
    new_line_to_be_divide=[]
    for i in range(len(line_to_be_divide)):
        ld = line_to_be_divide[i]
        P1 = ld[0]
        P2 = ld[1]
        New_p1, New_p2, New_p3, New_p4 =divide_line([P1,P2],num_pts) 
        num_pts=num_pts+4
        new_pts=new_pts+[New_p1, New_p2, New_p3, New_p4]
        new_line=[[P1,New_p1],[New_p2,P2]]
        for j in range(len(new_line)):
            line_segments.append(new_line[j])
        new_line_tb_div=[[New_p1,New_p3],[New_p3,New_p4],[New_p4,New_p2]]
        for j in range(len(new_line_tb_div)):
            new_line_to_be_divide.append(new_line_tb_div[j])
            
    return koch_snowflake(new_pts,num_pts,line_segments,new_line_to_be_divide, level - 1)

# define number of levels here
def MakeGeometry(fractal_level):
    geo = SplineGeometry()
    # add points
    p1=[np.array([0,1]),0]
    p2=[np.array([1,1]),1]
    p3=[np.array([1,0]),2]
    p4=[np.array([0,0]),3]
    geo.AppendPoint(*p1[0])
    geo.AppendPoint(*p2[0])
    geo.AppendPoint(*p3[0])
    geo.AppendPoint(*p4[0])
    id_pts=3

    # top snowflake
    new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p1,p2]], fractal_level)
    [geo.AppendPoint(*np[0]) for np in new_pts]
    line_list=line_list+line_list2
    [geo.Append (["line", L[1][1], L[0][1]], bc = 1) for L in line_list]
  
    # right side snowflake
    new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p2,p3]], fractal_level)
    [geo.AppendPoint(*np[0]) for np in new_pts]
    line_list=line_list+line_list2
    [geo.Append (["line", L[1][1], L[0][1]], bc = 2) for L in line_list]
 
    # bottom snowflake
    new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p3,p4]], fractal_level)
    [geo.AppendPoint(*np[0]) for np in new_pts]
    line_list=line_list+line_list2
    [geo.Append (["line", L[1][1], L[0][1]], bc = 3) for L in line_list]

    # left side snowflake
    new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p4,p1]], fractal_level)
    [geo.AppendPoint(*np[0]) for np in new_pts]
    line_list=line_list+line_list2
    [geo.Append (["line", L[1][1], L[0][1]], bc = 4) for L in line_list]
    
    # Add interior boundary
    pnt1_int=[(0.4,0.6)]
    pnt2_int=[(0.6,0.6)]
    pnt3_int=[(0.6,0.4)]
    pnt4_int=[(0.4,0.4)]
    p1_int=geo.AppendPoint(*pnt1_int)
    p2_int=geo.AppendPoint(*pnt2_int)
    p3_int=geo.AppendPoint(*pnt3_int)
    p4_int=geo.AppendPoint(*pnt4_int)
    # top
    geo,Append(["line",p1_int,p2_int],bc=5)
    # right
    geo,Append(["line",p2_int,p3_int],bc=6)
    # bottom
    geo,Append(["line",p3_int,p4_int],bc=7)
    # left
    geo,Append(["line",p4_int,p1_int],bc=8)
    return geo
