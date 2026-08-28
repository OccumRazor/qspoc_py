from . import propagation_method,Newton,test_utils,J_T_local
import numpy as np,copy
from scipy.linalg import expm
H = np.array([[1,-1j],[1j,0]])
dt = 0.1
psi = -1 * (1j)**2 * np.array([[1],[1]]) / np.sqrt(2)
psi = np.array([[1],[0]],dtype=np.complex128)
def func(z):
    return expm(-1j*z)


H = np.array([[0,0,0,-1j],[0,0,-1j,0],[0,1j,0,0],[1j,0,0,0]])
#H = np.array([[0,-1j],[1j,0]])
#H = np.eye(4,dtype=np.complex128)
psi = np.zeros([4,1],dtype=np.complex128)
psi[0,0] = 1
n = 16
dt = .4
def random_Herm(n):
    mat_0 = np.random.rand(n,n) + 1j * np.random.rand(n,n)
    return mat_0 + np.transpose(np.conjugate(mat_0))

def random_state(n):
    psi = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    return psi / np.linalg.norm(psi,2)
infidelity = np.zeros(100)
for i in range(100):
    H = test_utils.random_Herm(n)
    psi = test_utils.random_state(n)

    psi_T = Newton.Newton(copy.deepcopy(psi),H,dt,func)
    #print(psi_T)
    psi_t = propagation_method.Matrix_Exponential(H,psi,dt)
    #print(psi_t)
    infidelity[i] = J_T_local.JT_ss([psi_T],[psi_t])
    #print(J_T_local.JT_ss([psi_T],[psi_t]))
print(infidelity)
print(max(np.abs(infidelity)))