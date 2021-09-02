import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
import statsmodels.tsa.stattools as smt
from scipy.fftpack import fft
from fun_com_butterworth import rever_fun_1, rever_fun_2_poly,rever_fun_3

# 正弦波叠加法之经典谱的实现
# 里面包括了六种仿真平坦衰落信道的方法- 等距离法、等面积法、 蒙特卡洛法、最小均方误差法、精确多普勒扩展法和Jakers仿真法
class GenerateDopplerParas:
    # 初始化共有参数
    def __init__(self, Method_type='MEA',N_i=20,Variance=1,fd=100,phase='rand'):
        self.Method_type = Method_type
        self.N_i = N_i
        self.Var = Variance
        self.fd = fd
        self.phase = phase
        self.sigma = np.sqrt(Variance)
    # 生成 butterworth 参数

    def parameter_classical(self):
        # 初始化
        fmax = self.fd

        f_i = np.empty(self.N_i)
        c_i = np.empty(self.N_i)
        p_i = np.empty(self.N_i)

        # 生成固定的系数
        if self.Method_type == 'MED':
            n = np.arange(1, self.N_i + 1)
            f_i = fmax / (2 * self.N_i) * (2 * n - 1)
            c_i = (2 * self.sigma / np.sqrt(np.pi)) * np.sqrt(np.arcsin(n / self.N_i) - np.arcsin((n - 1) / self.N_i))
        elif self.Method_type == 'MEA':
            n = np.arange(1, self.N_i + 1)
            f_i = fmax * np.sin(np.pi * n / 2 / self.N_i)
            c_i = self.sigma * np.sqrt(2 / self.N_i) * np.ones(self.N_i)
        elif self.Method_type == 'MCM':
            n = np.random.rand(self.N_i)
            f_i = fmax * np.sin(np.pi * n / 2)
            c_i = self.sigma * np.sqrt(2 / self.N_i) * np.ones(self.N_i)
        elif self.Method_type == 'MSEM':
            n = np.arange(1, self.N_i + 1)
            f_i = fmax * (2 * n - 1) / (2 * self.N_i)
            T = 1 / (2 * fmax / self.N_i)
            M = 5e3
            tau = np.arange(0, T, 1 / M)
            Jo = spl.jv(0, 2 * np.pi * fmax * tau)

            c_i = []
            for m in range(self.N_i):
                c_i.append(2 * self.sigma * np.sqrt(1 / T * (np.trapz(tau, Jo * np.cos(2 * np.pi * f_i[m] * tau)))))

        elif self.Method_type == 'MEDS':
            n = np.arange(2, self.N_i + 2)
            f_i = fmax * np.sin(np.pi * (n - 0.5) / (2 * self.N_i))
            c_i = self.sigma * np.sqrt(2 / self.N_i) * np.ones(self.N_i)

        # 生成相位
        if self.phase == 'rand':
            p_i = 2 * np.pi * np.random.rand(self.N_i)

        return f_i, c_i, p_i


    def parameter_butterworth(self,r=1):
         f_i = np.array([],dtype=np.float64)
         c_i = np.empty(self.N_i,dtype=np.float64)
         p_i = np.empty(self.N_i,dtype=np.float64)
         pi = np.pi
    # 生成固定的系数
         if self.Method_type == 'MEA' and r == 1 :
              # 直接法
              # n = np.arange(1, N_i + 1)
              # c_i = sigma * np.sqrt(2/N_i) * np.ones(N_i)
              # f_i = fd * np.tan(pi*n/(4*N_i))
              n = np.arange(1, self.N_i + 1)
              c_i = self.sigma * np.sqrt(2 / self.N_i) * np.ones(self.N_i)
              for i in range(self.N_i):
                   f_i = np.append(f_i, self.fd * rever_fun_1(pi * (self.N_i + n[i]) / 4 / self.N_i))

         elif self.Method_type == 'MEA' and r == 2 :
              n = np.arange(1, self.N_i + 1)
              c_i = self.sigma * np.sqrt(2 / self.N_i) * np.ones(self.N_i)
              for i in range(self.N_i):
                   f_i = np.append(f_i,self.fd*rever_fun_2_poly(pi*(self.N_i+n[i])/4/self.N_i))

         elif self.Method_type == 'MEA' and r == 3 :
              n = np.arange(1, self.N_i + 1)
              c_i = self.sigma * np.sqrt(2 / self.N_i) * np.ones(self.N_i)
              for i in range(self.N_i):
                   f_i = np.append(f_i,self.fd*rever_fun_3(pi*(self.N_i+n[i])/4/self.N_i))

    # 生成相位
         if self.phase == 'rand':
              p_i = 2*np.pi* np.random.rand(self.N_i)
         return f_i,c_i,p_i


    def Butterworth_Belta(ci,fi,var):
         return np.sqrt(sum(np.power(ci,2)*np.power(fi,2))/2/var)

    def main(self):
        f1, c1, p1 = self.parameter_butterworth()
        f_max = self.fd
        Tau_max = (self.N_i / 2 / f_max) * 10  # 总时长
        Tau_int = 1 / (10 * f_max)  # 采样频率
        Tau = np.arange(0, Tau_max, Tau_int)  # 采样时刻
        N = int(Tau_max / Tau_int)  # 采样点数（采样的序列的长度）
        Gaussian_Procss = np.array([], dtype=np.float64)
        for i in Tau:
            Gaussian_Procss = np.append(Gaussian_Procss, sum(c1 * np.cos(2 * np.pi * f1 * i + p1)))
        plt.figure()
        plt.plot(Tau,Gaussian_Procss)
        plt.show()


if __name__ == '__main__':
    Params = GenerateDopplerParas()
    Params.main()
