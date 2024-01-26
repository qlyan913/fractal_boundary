# plot figures of results
import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict

nn=1
# dimension of fractal boundary
dim_frac=np.log(20)/np.log(4)
l=(1/4.)**nn
Lp=(6./4.)**nn

# plot for 0<Lambda<1000
# load data
columns = defaultdict(list) # each value in each column is appended to a list
with open(f'results/Phi_Lam_{nn}.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(float(v)) # append the value into the appropriate list
LL = np.array(columns['Lambda'])
Phi = np.array(columns['flux'])
Phi0=Phi[0]
LL=LL[1:]
Phi=Phi[1:]
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
phi_2_log=np.log(phi_2)
alpha=1/(dim_frac/2)
plt.loglog(LL[0:-2], phi_2[0:-2],marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[-1]**alpha)*(phi_2[-1]),color='red',linestyle='dashed',linewidth=0.8)
alpha=1
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='black',linestyle='dashed',linewidth=0.8)
plt.loglog(LL,(LL)**alpha/(LL[-3]**alpha)*(phi_2[-3]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.legend(['$1/\Phi-1/\Phi_0$','$O(\Lambda^{2/d})$', '$O(\Lambda^{1})$'])
plt.xticks([10**(-4),10**(-3),10**(-2),10**(-1),1,10**(1),10**(2),10**3,l, Lp], ['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','l', 'Lp'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}.png")
print(f"Plot for 0<Lambda<1000 saved to figures/Phi_Lam_{nn}.png ")

# plot for Region 1: 0<Lambda <l
# load data
columns = defaultdict(list) # each value in each column is appended to a list
with open(f'results/Phi_Lam_{nn}_R1.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(float(v)) # append the value into the appropriate list
LL = np.array(columns['Lambda'])
Phi = np.array(columns['flux'])
LL=LL[1:]
Phi=Phi[1:]
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[-1]**alpha)*(phi_2[-1]),color='black',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.xticks([10**(-5),10**(-4),10**(-3),l],['$10^{-5}$','$10^{-4}$','$10^{-3}$','l'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{1})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R1.png")
print(f"plot for 0<Lambda<l saved to figures/Phi_Lam_{nn}_R1.png ")


# plot for Region 2: l<Lambda <L_p
# load data
columns = defaultdict(list) # each value in each column is appended to a list
with open(f'results/Phi_Lam_{nn}_R2.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(float(v)) # append the value into the appropriate list
LL = np.array(columns['Lambda'])
Phi = np.array(columns['flux'])
LL=LL[1:]
Phi=Phi[1:]
fig, axes = plt.subplots()
phi_2=[]
for i in range(len(Phi)):
   phi_2.append(1/Phi[i]-1/Phi0)
alpha=1/(dim_frac/2)
plt.loglog(LL, phi_2,marker='o',color='blue')
plt.loglog(LL,(LL)**alpha/(LL[0]**alpha)*(phi_2[0]),color='red',linestyle='dashed',linewidth=0.8)
plt.axvline(x=l,color='cyan',linestyle='dashed')
plt.axvline(x=Lp,color='cyan',linestyle='dashed')
plt.xticks([10**(-2),10**(-1),1,l, Lp], ['$10^{-2}$','$10^{-1}$','$10^{0}$','l','Lp'])
plt.legend(['$1/\Phi-1/\Phi_0$', '$O(\Lambda^{2/d})$'])
plt.xlabel('$\Lambda$')
plt.savefig(f"figures/Phi_Lam_{nn}_R2.png")
print(f"Plot for l<Lambda<L_p saved to figures/Phi_Lam_{nn}_R2.png ")

# plot for Region 3: Lp< Lambda<infty
# load data
columns = defaultdict(list) # each value in each column is appended to a list
with open(f'results/Phi_Lam_{nn}_R2.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(float(v)) # append the value into the appropriate list
LL = np.array(columns['Lambda'])
Phi = np.array(columns['flux'])
LL=LL[1:]
Phi=Phi[1:]
fig, axes = plt.subplots()
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
print(f"Plot for Lp<Lambda<infty saved to figures/Phi_Lam_{nn}_R3.png ")


