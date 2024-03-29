from Ex3_solver import *
d_ins = np.linspace(0,7,20)
n=4
dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in d_ins]
dy_list_log=np.log(dy_list)
uu=np.zeros(len(dy_list),dtype=float)
for i in range(len(uu)):
   uu[i]=10**(-7)*dy_list[i]**1.05
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
