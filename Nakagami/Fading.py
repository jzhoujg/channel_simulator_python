from generate_classical import parameter_classical
import numpy as np
import matplotlib.pyplot as plt



# 高斯的系数

var = 1
N_i = 20
f_max = 100

u1 = parameter_classical('MEA',N_i,var,f_max,'rand')
u2 = parameter_classical('MEA',N_i,var,f_max,'rand')
u3 = parameter_classical('MEA',N_i,var,f_max,'rand')
