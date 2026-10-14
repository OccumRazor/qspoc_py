import numpy as np
from .localTools import random_guess,control_generator_random

def gen_pulse_option(oct_lambda_a,tlist,H_t,n_ctrl):
    pulse_options={}
    for i in range(1,n_ctrl + 1):
        this_pulse = control_generator_random(1,.2,tlist[1])[0]
        pulse_options[H_t[i][1]]=dict(oct_lambda_a = oct_lambda_a, t_rise = 0.01, t_fall = 0.01, args=this_pulse,
                                                oct_shape = 'flattop')
    return pulse_options

def random_Herm(n):
    mat_0 = np.random.rand(n,n) + 1j * np.random.rand(n,n)
    return mat_0 + np.transpose(np.conjugate(mat_0))

def random_state(n):
    psi = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    return psi / np.linalg.norm(psi,2)

def random_Hamiltonian(n,n_Herm,T):
    H_t = [random_Herm(n)]
    H_t += [[random_Herm(n),lambda t,args:random_guess(t,args)] for _ in range(n_Herm - 1)]
    return H_t, gen_pulse_option(0,[0,T,101],H_t,n_Herm-1)

def random_Lindbladian(n,n_Herm,n_c_ops,T):
    H = random_Hamiltonian(n,n_Herm,T)
    return H, [random_Herm(n) for _ in range(n_c_ops)]