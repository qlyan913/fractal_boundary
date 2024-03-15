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
resultsfile = 'results/Phi_Lam_3.csv'
# file for graphical output
plotfile = 'plot_Z_3D_3.png'

### ADJUST COMMENT
# read the data for Phi and Lambda.  Here for a square with a Koch snowflake prefractal top
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
nn=3  # pre-fractal generation
l=(1/4)**nn  # smallest length scale
Lp=(6/4)**nn # length/area of fractal ??
dim_frac=np.log(20)/np.log(4)  # fractal dimension
pw = 2./dim_frac   #1/dim_frac in 2D, 2/dim_frac in 3D

# set coefficients so that the power law approximation is exact at the points Lambda nearest
# 0 and infinity
c1 = z[2]/lam[2]  # lam[0] may be zero, so we use the second lowest lambda to fit in the near Dirichlet range
c2 = z[-1]/lam[-1]
p = np.exp((np.log(l) + np.log(Lp))/2)
ind = np.argmin((lam - p)**2)
c3 = z[ind]/lam[ind]**pw

### ADJUST
# plot limits
xmin = 1.e-4
xmax = 1.e+4
ymin = 1.e-4
ymax = 1.e+4
# annotation placement
textlx = 1.1e-3
textly = 3.e+3
textLx = 5.e+0
textLy = 3.e+3

# vertical lines at l and L_p
plt.loglog([l, l], [ymin, ymax], linewidth=6, color='lavender', alpha=0.8)
plt.text(textlx, textly, '$\\Lambda=\ell$')
plt.text(textLx, textLy, '$\\Lambda=L_p$')
plt.loglog([Lp, Lp], [ymin, ymax], linewidth=6, color='lavender', alpha=0.8)
# plot computed values of Z and comparison lines
plt.loglog(lam[1:], c1*lam[1:], label= f'${c1:.2f}\,\Lambda$', linewidth=2, color='goldenrod', alpha=0.75)
plt.loglog(lam[1:], c2*lam[1:], label= f'${c2:.2f}\,\Lambda$', linewidth=2, color='goldenrod', alpha=0.75)
plt.loglog(lam[1:], c3*lam[1:]**(pw), label=f'${c3:.2f}\,\Lambda^{{ {pw:.2f} }}$', linewidth=2, color='yellowgreen')  
plt.loglog(lam[1:], z[1:], 'o', label='$Z(\Lambda)$', markersize=6, alpha=.8)
plt.legend(fontsize=12)
plt.xlabel('$\Lambda$')
plt.ylabel('$Z(\Lambda$)')

### ADJUST
# the title has to be adjusted for the actual problem parameters (nn, l, L_p, fractal dimension),
# or use a different title, or none at all
plt.title(f'$Z$ as a function of $\Lambda$ in square w/ Koch snowflake top, generation $n=5$.\n$\ell=1/4^n\\approx {1/4**nn:.4f},\\ L_p=(6/4)^n\\approx {(6/4)**nn:.2f},\\ d = \\log 20/\\log 4,\\ 2/d\\approx {2*np.log(4)/np.log(20):.2f}$.', fontsize=6)

# optional settings
### ADJUST: optional settings
squareoutput = False
grid = True
if squareoutput:
    plt.axis('square')
    ticklocs = [10.**i for i in range(-6, 7)]
    ticklabs = [ plt.Text(10.**i, 0, f'$\\mathdefault{{10^{{{i}}}}}$') for i in range(-6, 7) ]
    plt.xticks(ticklocs, ticklabs)
    plt.yticks(ticklocs, ticklabs)
if grid:
    plt.grid()

# Set ranges for axes
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

plt.savefig(plotfile, dpi=300)
print(f'plot exported to {plotfile}')
