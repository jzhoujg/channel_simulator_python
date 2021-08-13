import numpy as np
import matplotlib.pyplot as plt


var = 1
fmax =160
pi = np.pi

f = np.arange(-160,160,0.01)

S_uiui = var/(fmax*np.sqrt(1 - np.power(f/fmax,2)))

plt.figure()
plt.plot(f,S_uiui)
plt.title("Classical Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w")# 设置纵轴标签
plt.legend(loc="upper right")
plt.show()
