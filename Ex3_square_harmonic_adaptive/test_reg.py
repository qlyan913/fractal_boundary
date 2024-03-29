from Ex3_solver import *
d_ins = np.linspace(0,7,20)
n=4
dy_list=[0.5*(1/3.)**n*(1/2.)**i for i in d_ins]
dy_list_log=np.log(dy_list)
#uu=[3.42420472e-09,3.16283084e-09,2.74764362e-09,2.28700225e-09,1.84903288e-09,1.46678403e-09,1.14974975e-09, 8.94808225e-10, 6.93578838e-10, 5.36472549e-10,4.1456686e-10,3.20279116e-10, 2.47459943e-10,1.91247929e-10, 1.47853037e-10, 1.14342066e-10, 8.84529169e-11, 6.84434051e-11, 5.29720036e-11,4.10052676e-11]
uu=[4.6098275e-11, 4.33023083e-11, 3.73845945e-11, 3.0816977e-11, 2.47372375e-11, 1.95507814e-11, 1.53116436e-11, 1.1928404e-11, 9.26474635e-12,7.18390607e-12, 5.56552433e-12,4.30989144e-12,3.33695971e-12,2.58355996e-12, 2.0003143e-12, 1.54882629e-12, 1.19932177e-12, 9.28744959e-13,7.19253743e-13, 5.57043542e-13]
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
