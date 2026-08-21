from . import propagation_method,Newton
import numpy as np
from scipy.linalg import expm
H = np.array([[1,-1j],[1j,0]])
dt = 0.1
psi = -1 * (1j)**2 * np.array([[1],[1]]) / np.sqrt(2)
psi = np.array([[1],[0]],dtype=np.complex128)
def func(z):
    return expm(-1j*z)
from . import arnoldi

#psi = np.ones([5,1]) / np.sqrt(5)


H = np.array([[0,0,0,-1j],[0,0,-1j,0],[0,1j,0,0],[1j,0,0,0]])
#H = np.array([[0,-1j],[1j,0]])
#H = np.eye(4,dtype=np.complex128)
psi = np.zeros([4,1],dtype=np.complex128)
psi[0,0] = 1
psi_T = Newton.Newton(psi,H,dt,func)
print(psi_T)
psi_t = propagation_method.Matrix_Exponential(H,psi,dt)
print(psi_t)