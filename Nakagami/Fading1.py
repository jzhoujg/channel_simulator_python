from generate_classical import parameter_classical
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt
import scipy.special as spl
from numpy import power
import math

def Generate_Nakagami(m=1.25,var=1,N_i=20,f_max=100):
    # 根据m计算需要的高斯衰落的数量，在这里默认的是平坦衰落的过程
    ##  至少需要两个高斯
    ## q1为整数部分
    q_1 = int(math.floor(m))

    belta = m - q_1 + np.sqrt((q_1-m)*(m-q_1-1))
    gamma = m - q_1 - np.sqrt((q_1-m)*(m-q_1-1))

    # 生成高斯衰落
    num = 2000
    ## 生成整数部分
    if q_1:
        U_q = np.zeros([num,q_1*2])
        for i in range(2*q_1):
            U_q[:, i], _ = parameter_classical('MEA',N_i,var,f_max,'rand',num)

    ## 生成小数部分

    U_p = np.empty([num,2])
    U_p[:, 0], t = parameter_classical('MEA',N_i,var,f_max,'rand',num)
    U_p[:, 1], _ = parameter_classical('MEA',N_i,var,f_max,'rand',num)

    # 生成Nakagami 分布u
    ##  Brute Force 原始模型 m = 2
    # R = np.sqrt(power(u1,2)+power(u2,2)+power(u3,2)+power(u4,2))
    # acf_bru1 = smt.acf(R,nlags= len(t))

    ## Brute Force 相位修正模型 m = 2
    # for i in range(len(u1)):
    #     if u1[i] < 0 : R[i] = -R[i]
    # acf_bru2 = smt.acf(R,nlags= len(t))

    ## Brute Force 精确小数模型
    R = []
    if q_1:
        for i in range(num):
            part2 = belta*(U_p[i,0]**2) + gamma*(U_p[i,1]**2)
            R.append(np.sqrt(sum(np.power(U_q[i,:],2)))+part2)
    else:
        for i in range(num):
            R.append(belta*(U_p[i,0]**2) + gamma*(U_p[i,1]**2))

    # 相位修正模型
    for i in range(num):
        if U_p[i,0] < 0:
            R[i] = -R[i]

    return np.array(R),t


# 生成瑞利过程
print()



f_max = 100
var = 1
m = 0.3
naka, t = Generate_Nakagami(f_max=f_max, var=1, m=m)
acf = smt.acf(naka, nlags=len(t))
R_rei_theory = var * spl.jv(0,2*np.pi*f_max*t) # 理论的自相关函数
# 画图

plt.figure()
plt.plot(t, acf, label='Experiment')
plt.plot(t, R_rei_theory, color = 'red',linestyle = '--',label = 'Theory')
plt.title("Nakagami Fading ACF m = "+str(m))
plt.xlabel("/s")# 设置横轴标签
plt.ylabel("")# 设置纵轴标签
plt.legend(loc="upper right")
plt.show()