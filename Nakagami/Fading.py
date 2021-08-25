from generate_classical import parameter_classical
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt
import scipy.special as spl
from numpy import power

# 高斯的系数

var = 1
N_i = 20
f_max = 100

# 生成高斯衰落
u1,t= parameter_classical('MEA',N_i,var,f_max,'rand')
u2,t= parameter_classical('MEA',N_i,var,f_max,'rand')
u3,t = parameter_classical('MEA',N_i,var,f_max,'rand')
u4,t = parameter_classical('MEA',N_i,var,f_max,'rand')

# 生成瑞利过程
ray = u1 + u2*1j
acf = smt.acf(ray,nlags= len(t))
R_rei_theory =var * spl.jv(0,2*np.pi*f_max*t) # 理论的自相关函数

# 生成Nakagami 分布u
# Brute Force 原始模型 m = 2

R = np.sqrt(power(u1,2)+power(u2,2)+power(u3,2)+power(u4,2))
acf_bru1 = smt.acf(R,nlags= len(t))











# 画图
plt.figure()

plt.plot(t,acf_bru1,label = 'Experiment')
plt.plot(t,R_rei_theory,color = 'red',linestyle = '--',label = 'Theory')
plt.title("Nakagami Fading ACF_B1 ")
plt.xlabel("/s")# 设置横轴标签
plt.ylabel("")# 设置纵轴标签
plt.legend(loc="upper right")
plt.show()