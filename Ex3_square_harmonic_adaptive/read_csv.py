import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from netgen.geom2d import SplineGeometry
from geogen import *
n=int(input("Enter the number of refinement steps for the pre-fractal upper boundary: "))
N= int(input("Enter the number of segments for estimation on each sides of the bottom  boundary: "))
# get bottom lines
p3=[np.array([1,0]),1]
p4=[np.array([0,0]),2]
id_pts=2
new_pts,id_pts,line_list,line_list2=koch_snowflake([],id_pts,[],[[p3,p4]], n)
line_list=line_list+line_list2

columns = defaultdict(list) # each value in each column is appended to a list
with open(f'results/Results_n{n}_N{N}.csv') as f:
    fieldnames = ['x','y','alpha','std_alpha', 'c']
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list

pt_xlist=np.array(columns['x'])
pt_ylist=np.array(columns['y'])
alpha_list=np.array(columns['alpha'])
std_list=np.array(columns['std_alpha'])

fig=plt.figure()
indx_list=sort_index(line_list,pt_xlist,pt_ylist) # this function is in geogen.py
im=plot_colourline(pt_xlist,pt_ylist,alpha_list,indx_list)
fig.colorbar(im)
plt.savefig(f"figures/alpha_n{n}_N{N}.pdf")
plt.close()
print(f"plot of alpha with color is saved in figures/alpha_n{n}_N{N}.pdf")

fig=plt.figure()
indx_list=sort_index(line_list,pt_xlist,pt_ylist) # this function is in geogen.py
im=plot_colourline(pt_xlist,pt_ylist,std_list,indx_list)
fig.colorbar(im)
plt.savefig(f"figures/alpha_std_n{n}_N{N}.pdf")
plt.close()
print(f"plot of std of alpha with color is saved in figures/alpha_std_cm_n{n}_N{N}.pdf")


