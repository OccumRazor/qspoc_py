from . import propagation_method,Newton,test_utils,J_T_local,task_obj
import numpy as np,copy
from scipy.linalg import expm
H = np.array([[1,-1j],[1j,0]])
dt = 0.1
psi = -1 * (1j)**2 * np.array([[1],[1]]) / np.sqrt(2)
psi = np.array([[1],[0]],dtype=np.complex128)
def func(z):
    return expm(-1j*z)

def config_prop_jobs():
    n = 4
    T = .3
    tlist = np.linspace(0,T,101)
    tlist = [0,T,101]
    initial_states = [test_utils.random_state(n)]
    H_t, pulse_options = test_utils.random_Hamiltonian(n,3,T)
    prop = task_obj.Propagation(H_t,tlist,'Newton',initial_states,'pulse_initial',pulse_options)
    psi_T = prop.propagate()[0]
    print(psi_T)
    print(np.linalg.norm(psi_T,2))
    return prop

config_prop_jobs()

H = np.array([[0,0,0,-1j],[0,0,-1j,0],[0,1j,0,0],[1j,0,0,0]])
c_ops = [.2 * np.kron(np.eye(2),[[0,1],[1,0]])]
#H = np.array([[0,-1j],[1j,0]])
#H = np.eye(4,dtype=np.complex128)
psi = np.zeros([4,1],dtype=np.complex128)
psi[0,0] = 1
n = 4
dt = .4
def random_Herm(n):
    mat_0 = np.random.rand(n,n) + 1j * np.random.rand(n,n)
    return mat_0 + np.transpose(np.conjugate(mat_0))

def random_state(n):
    psi = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    return psi / np.linalg.norm(psi,2)
infidelity = np.zeros(100)
for i in range(1):
    H,pulse_options = test_utils.random_Hamiltonian(n,2,.2)
    psi = test_utils.random_state(n)
    prop = task_obj.Propagation(H,[.2,101],'Newton',[psi],pulse_options=pulse_options,c_ops=c_ops)
    psi_T = prop.propagate()
    for psi in psi_T:
        print(psi.reshape(n,n).T.diagonal())
        print(sum(psi.reshape(n,n).T.diagonal()))
    #psi_T = Newton.Newton(H,copy.deepcopy(psi),dt,func)
    #print(psi_T)
    #psi_t = propagation_method.Matrix_Exponential(H,psi,dt)
    #print(psi_t)
    #infidelity[i] = J_T_local.JT_ss([psi_T],[psi_t])
    #print(J_T_local.JT_ss([psi_T],[psi_t]))
#print(infidelity)
#print(max(np.abs(infidelity)))