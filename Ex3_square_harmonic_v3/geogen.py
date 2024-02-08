from netgen.geom2d import SplineGeometry
import numpy as np
#  Create a mesh where each sides are replaced by snowflake
#  Boundary surfaces are numbered as follows:
#  boundary index: 1
def MakeGeometry():
    geo = SplineGeometry()

    pnt1=(0,0)
    pnt2=(1,0)
    pnt3=(1,1)
    pnt4=(0,1)
    P1=geo.AppendPoint(*pnt1)
    P2=geo.AppendPoint(*pnt2)
    P3=geo.AppendPoint(*pnt3)
    P4=geo.AppendPoint(*pnt4)
    geo.Append (["line", P1, P2], bc = 1)
    geo.Append (["line", P2, P3], bc = 1)
    geo.Append (["line", P3, P4], bc = 1)
    geo.Append (["line", P4, P1], bc = 1)
    return geo
