import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
import statsmodels.tsa.stattools as smt
from scipy.fftpack import fft
from rounded_fun import classical3dB_inverse

# 正弦波叠加法之经典谱的实现
# 里面包括了六种仿真平坦衰落信道的方法- 等距离法、等面积法、 蒙特卡洛法、最小均方误差法、精确多普勒扩展法和Jakers仿真法

def parameter_rounded(Method_type='MEA',N_i=20,Variance=1,fd=100,phase='rand') :
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
          for i in range(N_i):
               f_i = np.append(f_i,rounded_inverse(var/2*(1+n[i]/N_i),f_max))
# 生成相位
     if phase == 'rand':
          p_i = 2*np.pi* np.random.rand(N_i)

     return f_i,c_i,p_i

# 计算实际的多普勒扩展值
def Rounded_Belta(ci,fi,var=1):
     return np.sqrt(sum(np.power(ci,2)*np.power(fi,2))/2/var)


var = 1
N_i = 32
f_max = 100

# 生成一个高斯过程
f1,c1,p1 = parameter_rounded('MEA',N_i,var,f_max,'rand')

belta_cal = []

N = [i for i in range(5,50)]
for i in N:
    f1, c1, p1 = parameter_rounded(N_i=i)
    belta_cal.append(Rounded_Belta(c1,f1))
belta_cal = np.array(belta_cal)/f_max
belta_theory = np.ones(len(N)) * 41.6965/f_max

plt.figure()
# plt.stem(f1,c1/max(c1))
plt.plot(N,belta_cal,label = 'Experiment')
plt.plot(N,belta_theory,label = 'Theory',linestyle = '--')
plt.title("Belta")
plt.legend(loc="best",fontsize = 14)
plt.xlabel("N")# 设置横轴标签
plt.ylabel("")# 设置纵轴标签
plt.show()