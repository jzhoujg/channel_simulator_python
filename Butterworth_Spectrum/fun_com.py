from scipy.integrate import quad
from numpy import power
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc

def y1(x):
    return 1 / (power(x,2)+1)

def fun_1_order(x):
    return quad(y1,-1,x)[0]

def rever_fun_1(x):
    return inversefunc(fun_1_order,y_values=x)

def y2(x):
    return 1 / (power(x,4)+1)

def fun_2_order(x):
    return quad(y2,-1,x)[0]

def rever_fun_2(x):
    return inversefunc(fun_2_order,y_values=x)

def rever_fun_2_poly(x):
    return 0.3517 * power(x,5) - 1.525 * power(x,4) + 2.567 *power(x,3) -2.092 *power(x,2) + 1.828 *power(x,1)- 0.996


def y3(x):
    return 1 / (power(x,6)+1)

def fun_3_order(x):
    return quad(y3,-1.000000001,x-0.00001)[0]

def rever_fun_3(x):
    return inversefunc(fun_3_order,y_values=x)



# fd = 100
# N_i =30
# pi = np.pi
# n = np.arange(1, N_i + 1)
# f_i = np.array([])
# fi_1 = fd * np.tan(pi*n/(4*N_i))
# for i in range(N_i):
#     f_i = np.append(f_i, fd * rever_fun_1(pi * (N_i + n[i]) / 4 / N_i))
#
# plt.figure()
# plt.plot(n,fi_1)
# plt.plot(n,f_i)
#
# plt.show()
# print(fun_1_order(1))
# print(rever_fun_1(1.5707963267948968))
# f = np.arange(-1,1,0.001)
# res = []
#
# for i in f:
#     res.append(fun_2_order(i))


# plt.figure()
# plt.plot(f,res)
# plt.show()


