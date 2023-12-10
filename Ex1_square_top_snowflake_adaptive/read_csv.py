from firedrake.petsc import PETSc
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
columns = defaultdict(list) # each value in each column is appended to a list
nn=4
l=(1/3)**nn
Lp=(4/3)**nn
dim_frac=np.log(4)/np.log(3)

with open(f'results/Phi_Lam_{nn}.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list

LL=np.array(columns['Lambda'])
Phi=np.array(columns['flux'])
Phi0=Phi[0]

# R1: 0<Lambda <l
with open(f'results/Phi_Lam_{nn}_R1.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list
LL=np.array(columns['Lambda'])
Phi=np.array(columns['flux'])
Phi=Phi[1:]
LL=LL[1:]
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[-1]**alpha)*(phi_2[-1]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.xticks([10**(-9),10**(-8),10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),l],['$10^{-9}$','$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$','$10^{-4}$','$10^{-3}$','l'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R1.png")
PETSc.Sys.Print(f"plot for 0<Lambda<l saved to figures/Phi_Lam_{nn}_R1.png ")

# R2: l<Lambda <Lp
with open(f'results/Phi_Lam_{nn}_R2.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list
LL=np.array(columns['Lambda'])
Phi=np.array(columns['flux'])
Phi=Phi[1:]
LL=LL[1:]
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1/dim_frac
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='red',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.xticks([10**(-2),10**(-1),1,l, Lp], ['$10^{-2}$','$10^{-1}$','$10^{0}$','l','Lp'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1/dim_frac})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R2.png")
PETSc.Sys.Print(f"Plot for l<Lambda<L_p saved to figures/Phi_Lam_{nn}_R2.png ")

# R3:Lp<Lambda
with open(f'results/Phi_Lam_{nn}_R3.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list
LL=np.array(columns['Lambda'])
Phi=np.array(columns['flux'])
Phi=Phi[1:]
LL=LL[1:]
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.xticks([10**(1),10**(2),10**(3),10**(4),10**(5), Lp], ['$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','Lp'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R3.png")
PETSc.Sys.Print(f"Plot for Lp<Lambda<infty saved to figures/Phi_Lam_{nn}_R3.png ")
