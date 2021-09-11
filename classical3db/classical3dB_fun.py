import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pynverse import inversefunc
from numpy import sqrt

pi = np.pi

def classical3dB(f0,var=1,fd=100):
    f0 =f0/fd
    S_uiui = var * sqrt(var / (pi*fd*sqrt(1-np.power((f0),2))))/13.513239100813443
    return S_uiui

def classical3dB_quad(x,fd=100):
    return quad(classical3dB,-fd+0.01,x)[0]

def classical3dB_inverse(x):
    return inversefunc(classical3dB, y_values=x)

def classical3dB_inverse_poly(x):
    return -227.5*x**5 + 570.3 * x ** 4 - 664.9 * x ** 3 + 426 * x ** 2 + 97.5 * x - 100.6

def belta_cal(f):
    return (f**2) * classical3dB(f)*classical3dB

def belta_theory(fd):
    return quad(belta_cal,-fd,fd)[0]

f = np.arange(-100,100,0.5)
a = []

for i in range(len(f)):
    a.append(classical3dB_quad(f[i]))

z1 = np.polyfit(a,f,5)
d1 = np.poly1d(z1)
print(d1)
# print(classical3dB_quad(100))
# plt.figure()
# plt.plot(f,a)
# plt.show()


