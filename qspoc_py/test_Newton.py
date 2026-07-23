from . import propagation_method,Newton
import numpy as np
from scipy.linalg import expm
H = np.array([[1,-1j],[1j,0]])
dt = 2e-3
psi = -1 * (1j)**2 * np.array([[1],[1]]) / np.sqrt(2)
def func(z):
    return expm(-1j*z)
psi_T = Newton.Newton(psi,-1j*H,dt,func)
print(psi_T)
psi_t = propagation_method.Matrix_Exponential(H,psi,dt)
print(psi_t)