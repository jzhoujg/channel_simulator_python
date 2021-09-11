import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from numpy import sqrt

var = 1
fd = 50
pi = np.pi
f = np.arange(-fd,fd)
f0 = f/fd
S_uiui = sqrt(var / (pi*fd*sqrt(1-(f0)**2)))


plt.figure()
plt.plot(f,S_uiui)
plt.title("Classical 3dB Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w/Hz")# 设置纵轴标签
plt.legend(loc="best")
plt.show()