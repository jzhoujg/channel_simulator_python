import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

var = 1
fd = 40

# pi = np.pi
# a0 = 1
# a2 = -1.72
# a4 = 0.785
# Cr = 1 / (2*fd*(a0+a2/3+a4/5))

f = np.arange(-fd,fd+1)
# f0 = f/fd

S_uiui = np.zeros(len(f))

S_uiui[0] = var/2
S_uiui[len(f)-1] = var/2

plt.figure()
plt.stem(f,S_uiui,label = 'fd = ' + str(fd) + 'Hz')
plt.title("Pure Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w/Hz")# 设置纵轴标签
plt.legend(loc="best")
plt.show()