from Ex3_solver import *
d_ins = np.linspace(0,7,20)
n=4
dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in d_ins]
dy_list_log=np.log(dy_list)
uu=[3.42420472e-09,3.41682668e-09,3.22430482e-09,2.8771995e-09,2.38030998e-09,1.92918253e-09, 1.54276068e-09, 1.22268238e-09, 9.63013582e-10,7.55184242e-10,5.90355464e-10, 4.60453477e-10, 3.58534133e-10, 2.78826787e-10, 2.16637235e-10, 1.68199908e-10, 1.30522749e-10, 1.01244091e-10, 7.85086994e-11, 6.08642177e-11]
uu_log=np.log(uu)
alpha,b,std=std_linreg(dy_list_log, uu_log) 
c=exp(b)
#plot_reg("test_r.png",uu,c,dy_list,alpha)
def plot_reg(filename,uu,c,dy_list,alpha):
   tt=c*(dy_list)**alpha
   plt.figure()
   plt.loglog(dy_list,uu,'b.')
   plt.loglog(dy_list,tt)
   plt.ylabel('evaluation of solution')
   plt.xlabel('distance to boundary')
   plt.legend(['value of solution','${%s}(dx)^{{%s}}$' % (c,alpha)])
   plt.savefig(filename)
   plt.close()
   print(f"plot of one example of solution in loglog is saved in ", filename)

plot_reg("test_r.png",uu,c,dy_list,alpha)
