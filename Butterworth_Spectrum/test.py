import numpy as np
import fun_com as f

# xxx = np.arange(-1,1,0.001)
# yyy = np.array([])
#
# for i in xxx:
#     yyy = np.append(yyy,f.fun_2_order(i))
# z1 = np.polyfit(yyy,xxx,5)
# d1 = np.poly1d(z1)
#
#
# print(f.rever_fun_2(1),f.rever_fun_2_poly(1))
# print(d1)


print(f.fun_3_order(1)/f.fun_1_order(1))