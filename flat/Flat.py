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


S_uiui = 1/2/fd * np.ones(len(f))

plt.figure()
plt.plot(f,S_uiui,label = 'fd = ' + str(fd) + 'Hz')
plt.title("Flat Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w/Hz")# 设置纵轴标签
plt.legend(loc="best")
plt.show()