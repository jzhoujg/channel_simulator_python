import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
# from

var = 1
fd = 100
pi = np.pi
M = 3
f = np.arange(-fd,fd)
S_uiui = np.power(fd,2*M)/(np.power(fd,2*M)+np.power(f,2*M))

plt.figure()
plt.plot(f,S_uiui,label = 'M = 3')

plt.title("Butterworth Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w/Hz")# 设置纵轴标签
plt.legend(loc="upper right")

plt.show()