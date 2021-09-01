import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from pynverse import inversefunc

def rounded(f0,var=1,fd=100,a0=1,a2=-1.72,a4=0.785):
    f0 =f0/fd
    Cr = 1 / (2*fd*(a0+a2/3+a4/5))
    S_uiui = var * Cr * (a0 + a2*(f0**2) + a4*(f0**4))
    return S_uiui

def rounded_quad(x,fd=100):
    return quad(rounded,-fd,x)[0]

def rounded_inverse(x,fd):
    return inversefunc(rounded_quad, y_values=x)


