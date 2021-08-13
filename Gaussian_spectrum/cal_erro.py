import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
import statsmodels.tsa.stattools as smt

# 正弦波叠加法之经典谱的实现
# 里面包括了六种仿真平坦衰落信道的方法- 等距离法、等面积法、 蒙特卡洛法、最小均方误差法、精确多普勒扩展法和Jakers仿真法

def parameter_classical(Method_type,N_i,Variance,fc,phase,t) :
#初始化
     ln2 = np.log(2)
     kc = 2*np.sqrt(2/ln2)
     sigma = np.sqrt(Variance)
     fc = np.sqrt(ln2)*fc
     fmax = kc*fc
     f_i = np.empty(N_i)
     c_i = np.empty(N_i)
     p_i = np.empty(N   _i)

# 生成固定的系数
     if  Method_type == 'MED':
          n = np.arange(1, N_i + 1)
          f_i = fmax/(2 * N_i)*(2 * n -1)
          c_i = sigma*np.sqrt(2)*(np.sqrt(spl.erf(n*kc*np.sqrt(ln2)/N_i)-spl.erf((n-1)*kc*np.sqrt(ln2)/N_i)))
     elif Method_type == 'MEA':
          n = np.arange(1, N_i+1)
          f_i = fc/np.sqrt(ln2)*spl.erfinv(n/N_i)
          f_i[N_i-1] = fc/np.sqrt(ln2)*spl.erfinv(0.99999999)
          c_i = sigma * np.sqrt(2/N_i) * np.ones(N_i)

     # elif Method_type == 'MCM':
     #      n = np.random.rand(N_i)
     #      f_i = fmax*np.sin(np.pi*n/2)
     #      c_i = sigma * np.sqrt(2/N_i)*np.ones(N_i)
     # elif Method_type == 'MSEM':
     #      n = np.arange(1, N_i + 1)
     #      f_i = fmax*(2*n-1)/(2*N_i)
     #      T = 1/(2*fmax/N_i)
     #      M = 5e3
     #      tau = np.arange(0,T,1/M)
     #      Jo = spl.jv(0,2*np.pi*fmax*tau)
     #
     #      c_i = []
     #      for m in range(N_i):
     #           c_i.append(2 * sigma * np.sqrt(1/T*(np.trapz(tau,Jo*np.cos(2*np.pi*f_i[m]*tau)))))
     # elif Method_type == 'MEDS' :
     #      n = np.arange(2, N_i +2)
     #      f_i = fc / np.sqrt(ln2) * spl.erfinv((2*n -1)/(2*N_i))
     #      c_i = sigma * np.sqrt(2/N_i) * np.ones(N_i)
     #


# 生成相位
     if phase == 'rand':
          p_i = 2*np.pi* np.random.rand(N_i)
     u = []

     for i in t:
          temp = 0
          for f,c,p in zip(f_i,c_i,p_i):
               temp += c*np.cos(2*np.pi*f*i+p)
          u.append(temp)

     return f_i,c_i,p_i



def generate_erro(N_i,Method):
    ln2 = np.log(2)
    # 生成u1、u2
    delta_t = 0.01/100
    #theta_c = 40
    T_s = 0.097
    t = np.arange(0,T_s,delta_t)
    var = 1
    f_max = 91

    # 生成两个高斯过程
    f1,c1,p1 = parameter_classical(Method,N_i ,var,f_max,'rand',t)

    Tau_max = N_i/(np.sqrt(ln2)*f_max*2*np.sqrt(2/ln2)*2)
    Tau_int = 0.0001

    tau = np.arange(0,Tau_max,Tau_int,dtype = np.float64)



    # 生成 瑞利分布的自相关函数
    R_rei_exper = np.array([],dtype=np.float64)
    R_rei_theory =var * np.exp(-np.power((np.pi)*f_max*tau,2))
    for i in tau:
        R_rei_exper = np.append(R_rei_exper,sum(np.power(c1,2) * np.cos(2*np.pi*f1*i))/2)



    return np.trapz(tau,np.power((R_rei_theory-R_rei_exper),2))/Tau_max

a = generate_erro(10,'MED')
err_med = np.array([])
err_mea = np.array([])
# err_meds = []
# err_mcm = []
# err = []

# N = [2*i-1 for i in range(2,50)]
N = [2*i -1 for i in range(3,25)]
for i in N:
    err_med = np.append(err_med,generate_erro(i,'MED'))
    err_mea = np.append(err_mea,generate_erro(i,'MEA'))
    # err_mcm.append(generate_erro(i,'MCM'))
    # err_meds.append(generate_erro(i, 'MEDS'))

    # err.append(err_mcm)
    # err_mcm = []

# print(err)
# res = err[0]
# err = np.array(err)
# print(err)
# for k in range(1,10):
#     res += err[k]
#
# res = res /10
#


plt.figure()
plt.title("ERRO")
plt.grid()
plt.plot(N,err_med,label = 'MED',linewidth=2)
plt.plot(N,err_mea,label = 'MEA',linewidth=2)
# plt.plot(N,err_meds,label = 'MEDS',linewidth=2)
# plt.plot(N,res,label = 'MCM', linestyle = "--",color = 'blue')
plt.xlabel("N_i")# 设置横轴标签
plt.ylabel("err")# 设置纵轴标签
plt.legend(loc="upper right")

plt.show()