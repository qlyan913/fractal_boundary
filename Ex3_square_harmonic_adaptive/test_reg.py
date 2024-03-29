from Ex3_solver import *
d_ins = np.linspace(0,7,20)
n=4
dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in d_ins]
dy_list_log=np.log(dy_list)
uu=[3.42420472e-09,3.16283084e-09,2.74764362e-09,2.28700225e-09,1.84903288e-09,1.46678403e-09,1.14974975e-09, 8.94808225e-10, 6.93578838e-10, 5.36472549e-10,4.1456686e-10,3.20279116e-10, 2.47459943e-10,1.91247929e-10, 1.47853037e-10, 1.14342066e-10, 8.84529169e-11, 6.84434051e-11, 5.29720036e-11,4.10052676e-11]
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
