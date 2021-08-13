import matplotlib.pyplot as plt
import numpy as np
theta_0 = 1.0
theta_c  = 3.0



f = np.arange(-1,1,0.001)

Su1_f = theta_0 * (theta_c /(f *f + theta_c))

plt.figure()

plt.plot(f,Su1_f)


plt.show()

