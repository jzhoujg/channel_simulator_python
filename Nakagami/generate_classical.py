import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
import statsmodels.tsa.stattools as smt
from scipy.fftpack import fft

# 正弦波叠加法之经典谱的实现
# 里面包括了六种仿真平坦衰落信道的方法- 等距离法、等面积法、 蒙特卡洛法、最小均方误差法、精确多普勒扩展法和Jakers仿真法

def parameter_classical(Method_type,N_i,Variance,fmax,phase) :
#初始化
     sigma = np.sqrt(Variance)
     f_max = fmax
     fmax += (np.random.rand(1)-0.5)/10
     f_i = np.empty(N_i)
     c_i = np.empty(N_i)
     p_i = np.empty(N_i)

# 生成固定的系数
     if  Method_type == 'MED':
          n = np.arange(1, N_i + 1)
          f_i = fmax/(2 * N_i)*(2 * n -1)
          c_i = (2 * sigma/np.sqrt(np.pi))*np.sqrt(np.arcsin(n/N_i)-np.arcsin((n-1)/N_i))
     elif Method_type == 'MEA':
          n = np.arange(1, N_i + 1)
          f_i = fmax*np.sin(np.pi*n/2/N_i)
          c_i = sigma * np.sqrt(2/N_i) * np.ones(N_i)
     elif Method_type == 'MCM':
          n = np.random.rand(N_i)
          f_i = fmax*np.sin(np.pi*n/2)
          c_i = sigma * np.sqrt(2/N_i)*np.ones(N_i)
     elif Method_type == 'MSEM':
          n = np.arange(1, N_i + 1)
          f_i = fmax*(2*n-1)/(2*N_i)
          T = 1/(2*fmax/N_i)
          M = 5e3
          tau = np.arange(0,T,1/M)
          Jo = spl.jv(0,2*np.pi*fmax*tau)

          c_i = []
          for m in range(N_i):
               c_i.append(2 * sigma * np.sqrt(1/T*(np.trapz(tau,Jo*np.cos(2*np.pi*f_i[m]*tau)))))

     elif Method_type == 'MEDS' :
          n = np.arange(2, N_i +2)
          f_i = fmax * np.sin(np.pi * (n - 0.5)/(2*N_i))
          c_i = sigma * np.sqrt(2/N_i) * np.ones(N_i)


     t_int = 0.0001
     t_max = N_i/2/f_max
     t = np.arange(0,t_max,t_int)
# 生成相位
     if phase == 'rand':
          p_i = 2*np.pi* np.random.rand(N_i)

# 生成高斯序列
     u = []
     for i in t :
          u.append(sum(c_i*np.cos(2 * np.pi * f_i * i + p_i)))
     return np.array(u),t


def Experiment_Belta(ci,fi,var):
     return np.sqrt(sum(np.power(ci,2)*np.power(fi,2))/2/var)

# var = 1
# N_i = 20
# f_max = 100
# # 生成两个高斯过程
# f1,c1,p1 = parameter_classical('MEA',N_i,var,f_max,'rand')



# f2,c2,p2 = parameter_classical('MEA',N_i,var,f_max,'rand',t)
# u2 = generate_Gaussian_process(f2,c2,p2)
# # 检验生成的高斯过程
# u1_ave = sum(u1)*delta_t/0.19
# print(u1_ave)
# u1_var = sum(np.power(u1-u1_ave,2))*delta_t/0.19
# print(u1_var)
#
#
# Tau_max = N_i/2/f_max
# Tau_int = 0.001
#
# tau = np.arange(0,Tau_max,Tau_int)
# R_u1u1_tau = []
#
# # R_u1u1_tau = np.power(c1,2) * np.cos(2*np.pi*f1*tau)
# # for i in tau:
# #      R_u1u1_tau.append(sum(np.power(c1,2) * np.cos(2*np.pi*f1*i)/2))
# # E_ruiui = (1/Tau_max)*(sum(np.power(R_u1u1_tau[1:],2)*Tau_int)+np.power(R_u1u1_tau[0]-var,2)*Tau_int)
# # print(E_ruiui)
#
# # 生成 瑞利分布
# Xi = np.sqrt(np.power(u1,2)+np.power(u2,2))
# Xi_log10 = 20 * np.log10(Xi)
#
# # 生成 瑞利分布的自相关函数
# R_rei_exper = []
# R_rei_theory =var * spl.jv(0,2*np.pi*f_max*tau) # 理论的自相关函数
# for i in tau:
#      R_rei_exper.append(sum(np.power(c1,2) * np.cos(2*np.pi*f1*i))/2)
#
#
# print(np.trapz(tau,np.power((R_rei_theory-R_rei_exper),2))/Tau_max)
#
# # E_rei = 1/Tau_max*np.trapz(Tau,np.power(,2))
#
#
#
#
#
#
# # 评估模型的性能
# ## 功率谱画图
#
# # 理论自相关计算
# #r_theory = np.exp(-2 * np.pi * theta_c * t)
#
#
# # 画图
# plt.figure()
# # plt.plot(t,r_theory)
# # 典型的U形状功率谱
# #plt.plot([i for i in range(31)],c1)
#
# # 衰落图
# plt.plot(t,Xi_log10)
# plt.show()