import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
import statsmodels.tsa.stattools as smt
from scipy.fftpack import fft

# 正弦波叠加法之经典谱的实现
# 里面包括了六种仿真平坦衰落信道的方法- 等距离法、等面积法、 蒙特卡洛法、最小均方误差法、精确多普勒扩展法和Jakers仿真法

def parameter_rounded(Method_type,N_i,Variance,fd,phase) :
#初始化
     sigma = np.sqrt(Variance)
     f_i = np.array([],dtype=np.float64)
     c_i = np.empty(N_i,dtype=np.float64)
     p_i = np.empty(N_i,dtype=np.float64)
     pi = np.pi
# 生成固定的系数
     if Method_type == 'MEA' :
          n = np.arange(1, N_i + 1)
          c_i = sigma * np.sqrt(2 / N_i) * np.ones(N_i)
          f_i = var*(1/2/N_i+n/N_i)*fd


# 生成相位
     if phase == 'rand':
          p_i = 2*np.pi* np.random.rand(N_i)

     return f_i,c_i,p_i


# 计算多普勒扩展值
def Rounded_Belta(ci,fi,var):
     return np.sqrt(sum(np.power(ci,2)*np.power(fi,2))/2/var)

var = 1
N_i = 32
f_max = 100

# 生成一个高斯过程
f1,c1,p1 = parameter_rounded('MEA',N_i,var,f_max,'rand')
# plt.figure()
# plt.stem(f1,c1/max(c1))
# plt.title('psd')
# plt.legend(loc="best",fontsize = 14)
# plt.xlabel("frequency/Hz")# 设置横轴标签
# plt.ylabel("p/w/Hz")# 设置纵轴标签
# plt.show()
#
# print(Butterworth_Belta(c1,f1,var),52.27232008770633)
# # 采样
# Tau_max = (N_i/2/f_max)*10 # 总时长
# Tau_int = 1/(10*f_max) # 采样频率
# Tau = np.arange(0,Tau_max,Tau_int) # 采样时刻
# N = int(Tau_max/Tau_int) # 采样点数（采样的序列的长度）
#
# # 采样的序列
# Gaussian_Procss = np.array([],dtype=np.float64)
# for i in Tau:
#      Gaussian_Procss = np.append(Gaussian_Procss,sum(c1*np.cos(2*np.pi*f1*i+p1)))
# L = len(Gaussian_Procss) # 信号长度
#
# #N = np.power(2,np.ceil(np.log2(L))) # 下一个最近二次幂
# res = np.empty(1,dtype=np.float64)
# res = abs(fft(Gaussian_Procss,L))
# h = np.array([i for i in range(int(N))])
# # plt.psd(Gaussian_Procss,NFFT=L)
# plt.figure()
# plt.plot(h[:int(N)]*10*f_max/N,res[:int(N)]/max(res), label = 'M = 1')
# plt.title('psd')
# plt.legend(loc="best",fontsize = 14)
# plt.show()
#
#
#
#
#



# tau = np.arange(0,Tau_max,Tau_int)
# R_u1u1_tau = []
# R_u1u1_tau = np.power(c1,2) * np.cos(2*np.pi*f1*tau)
# for i in tau:
#      R_u1u1_tau.append(sum(np.power(c1,2) * np.cos(2*np.pi*f1*i)/2))
# E_ruiui = (1/Tau_max)*(sum(np.power(R_u1u1_tau[1:],2)*Tau_int)+np.power(R_u1u1_tau[0]-var,2)*Tau_int)
# print(E_ruiui)
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