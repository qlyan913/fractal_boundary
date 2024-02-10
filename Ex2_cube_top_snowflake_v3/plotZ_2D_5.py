# This file creates a plot (as a png file) for plotting Z (normalized reciprocal flux) in terms
# of Lambda, and compares that to certain power law fits.
# User must adjust the lines flagged with '### ADJUST'

import numpy as np
import matplotlib.pyplot as plt
plt.clf()

### ADJUST
# resultsfile should be a csv file with 2 columns, one for Lambda, one for Phi,
# with 1st row for column labels.  The data is sorted by increasing Lambda and
# the least Lambda, typically Lambda, is not used for making this plot
# Data from Qile Yan, fractal_boundary/Ex1_square_top_snowflake_adaptive.
# plotfile is for the output
resultsfile = './Phi_Lam_3D_3.csv'
plotfile = './plot_Z_3D_3.png'

### ADJUST COMMENT
# read the data for Phi and Lambda.  Here for a square with asymmetric square Koch snowflake fractal top
# of generation 5.  Data from Qile Yan, fractal_boundary/Ex1_square_top_snowflake_adaptive
[lam, phi] = np.loadtxt(resultsfile, skiprows=1, delimiter=',').transpose()

# reorder the data so that Lambda is increasing
i = np.argsort(lam)
lam = lam[i]
phi = phi[i]
# define z
z = 1./phi - 1./phi[0]

### ADJUST
# geometric parameters
nn=5  # pre-fractal generation
l=(1/3)**nn  # smallest length scale
Lp=(4/3)**nn # length/area of fractal ??
dim_frac=np.log(4)/np.log(3)  # fractal dimension
pw = 1./dim_frac   #1/dim_frac in 2D, 2/dim_frac in 3D

# set coefficients so that the power law approximation is exact at the points Lambda nearest
# 0 and infinity
c1 = z[1]/lam[1]  # lam[0] may be zero, so we use the second lowest lambda to fit in the near Dirichlet range
c2 = z[-1]/lam[-1]
p = np.exp((np.log(l) + np.log(Lp))/2)
ind = np.argmin((lam - p)**2)
c3 = z[ind]/lam[ind]**pw

### ADJUST
# vertical lines at l and L_p
plt.loglog([l, l], [10.**(-11), 10**8], linewidth=6, color='lavender', alpha=0.8)
plt.text(4.e-4, 1.e-7, '$\\Lambda=\ell$')
plt.text(7., 1.e-7, '$\\Lambda=L_p$')
plt.loglog([Lp, Lp], [10.**(-9), 10**5], linewidth=6, color='lavender', alpha=0.8)

plt.loglog(lam[1:], c1*lam[1:], label= f'${c1:.2f}\,\Lambda$', linewidth=2, color='goldenrod', alpha=0.75)
plt.loglog(lam[1:], c2*lam[1:], label= f'${c2:.2f}\,\Lambda$', linewidth=2, color='goldenrod', alpha=0.75)
plt.loglog(lam[1:], c3*lam[1:]**(pw), label=f'${c3:.2f}\,\Lambda^{{ {pw:.2f} }}$', linewidth=2, color='yellowgreen')  
plt.loglog(lam[1:], z[1:], 'o', label='$Z(\Lambda)$', markersize=6, alpha=.8)
plt.legend(fontsize=12)
plt.xlabel('$\Lambda$')
plt.ylabel('$Z(\Lambda$)')

### ADJUST
# the title has to be adjusted for the actual problem parameters (nn, l, L_p, fractal dimension)
plt.title(f'$Z$ as a function of $\Lambda$ in asymmetric Koch snowflake, generation $n=3$.\n$\ell=1/4^n\\approx {1/3**nn:.4f},\\ L_p=(6/4)^n\\approx {(6/4.)**nn:.2f},\\ \\dim = \\log 20/\\log 4,\\ 1/\\dim\\approx {np.log(4)/np.log(20):.2f}$.', fontsize=10)

#plt.axis('square')
plt.grid()

### ADJUST
# Set ranges for axes
plt.xlim([10.**(-8), 10.**4.5])
plt.ylim([10.**(-8), 10.**5])

plt.savefig(plotfile, dpi=300)
print(f'plot exported to {plotfile}')

