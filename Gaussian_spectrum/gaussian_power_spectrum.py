import numpy as np
import matplotlib.pyplot as plt


var = 1
fc =50
pi = np.pi
ln2 = np.log(2)

f = np.arange(-160,160,0.01)

S_uiui = (var/fc)*np.sqrt(ln2/pi)*np.exp(-ln2*np.power(f/fc,2))

plt.figure()
plt.plot(f,S_uiui)
plt.title("Gaussian Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w")# 设置纵轴标签
plt.legend(loc="upper right")
plt.show()
