from . import propagation_method,Newton
import numpy as np
from scipy.linalg import expm
H = np.array([[1,-1j],[1j,0]])
psi = np.array([[1],[1]]) / np.sqrt(2)
psi_T = Newton.Newton(psi,-1j*H,1,expm)
print(psi_T)
psi_t = propagation_method.Matrix_Exponential(H,psi,1)
print(psi_t)