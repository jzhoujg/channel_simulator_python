import numpy as np
import matplotlib.pyplot as plt


var = 1


pi = np.pi
ln2 = np.log(2)
fc = 50*np.sqrt(ln2)
kc = 2*np.sqrt(2/ln2)
fl = fc* kc
f = np.arange(-fl,fl,0.01)

S_uiui = (var/fc)*np.sqrt(ln2/pi)*np.exp(-ln2*np.power(f/fc,2))

plt.figure()
plt.plot(f,S_uiui)
plt.title("Gaussian Doppler Spectrum ")
plt.xlabel("frequency/Hz")# 设置横轴标签
plt.ylabel("p/w")# 设置纵轴标签
plt.legend(loc="upper right")
plt.show()
