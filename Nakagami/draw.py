import matplotlib.pyplot as plt
import numpy as np

k = np.arange(0,10,0.001)
res = np.power((1+k),2)/(2*k+1)
plt.figure()
plt.plot(k,res)
plt.show()