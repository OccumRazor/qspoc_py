import numpy as np

def random_Herm(n):
    mat_0 = np.random.rand(n,n) + 1j * np.random.rand(n,n)
    return mat_0 + np.transpose(np.conjugate(mat_0))

def random_state(n):
    psi = np.random.rand(n,1) + 1j * np.random.rand(n,1)
    return psi / np.linalg.norm(psi,2)

