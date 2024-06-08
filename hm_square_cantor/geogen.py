from netgen.geom2d import SplineGeometry
import numpy as np
#  Create a mesh on sqaure with Cantor set inside
#  The domain is \Omega=Q\Q_n where Q=[-1,1] x [-1, 1] and Q_n is the nth iteration of 1D cantor set inside Q.
#  Boundary index are numbered as follows:
#  1: exterior boundary of square
#  2: interior boundary of square
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def MakeGeometry(cantor_level):
    # Input: number of fractal level
    geo = SplineGeometry()
    # Outside  boundary
    pnt1=(-1,-1)
    pnt2=(2,-1)
    pnt3=(2,2)
    pnt4=(-1,2)
    P1=geo.AppendPoint(*pnt1)
    P2=geo.AppendPoint(*pnt2)
    P3=geo.AppendPoint(*pnt3)
    P4=geo.AppendPoint(*pnt4)
    geo.Append (["line", P1, P2], bc = 1)
    geo.Append (["line", P2, P3], bc = 1)
    geo.Append (["line", P3, P4], bc = 1)
    geo.Append (["line", P4, P1], bc = 1)
    
    # Inside boundary
    line_list=cantor_line([[0.0,1.0]],cantor_level)
    for i in range(len(line_list)):
        x1=line_list[i][0]
        x2=line_list[i][1]
        pnt1=(x1,0.0)
        pnt2=(x2,0.0)
        P1=geo.AppendPoint(*pnt1)
        P2=geo.AppendPoint(*pnt2)
        geo.Append (["line", P1, P2], bc = 2)
    return geo

def divide_line(vertices):
    # input
    # vertices - array of shape 2x3.
    #          - two points P1 P2
    # out put
    # array of shape 2x3. Two points N1 and N2 on the line P1P2.
    #
    #  P1 ---- N1 ----- N2 -----P2
    Rot=np.array([[.5, -np.sqrt(3.)/2],[np.sqrt(3.)/2, .5]]) # rotation matrix
    P1 = vertices[0]
    P2 = vertices[1]
    dx = P2-P1
    N1 = P1+dx/3.
    N2 = N1+dx/3.
    New_p1=N1
    New_p2=N2
    return New_p1,New_p2

def cantor_line(line_list, level):
    # pts:           list of points, each element consists of position of the point and the index of the point.
    #                P1--------->---------P2
    # line_list: list of lines in 1D cantor set
    if level == 0:
        return line_list
    new_line_list=[]
    for i in range(len(line_list)):
        ld = line_list[i]
        P1 = ld[0]
        P2 = ld[1]
        New_p1, New_p2 = divide_line([P1,P2])
        new_line=[[P1,New_p1],[New_p2,P2]]
        for j in range(len(new_line)):
            new_line_list.append(new_line[j])
            
    return cantor_line(new_line_list, level - 1)

