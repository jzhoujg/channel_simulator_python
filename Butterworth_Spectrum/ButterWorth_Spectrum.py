import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
# from

var = 1
fd = 91
pi = np.pi
f = np.arange(-fd,fd)
S_uiui = np.power(fd,4)/(np.power(fd,4)+np.power(f,4))
plt.figure()
plt.plot(f,S_uiui)
plt.title("Gaussian Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w")# 设置纵轴标签
plt.legend(loc="upper right")
plt.show()