import numpy as np,copy
from scipy.linalg import expm
from . import arnoldi


def normalizaiton(Z:list):
    '''
    Docstring for normalizaiton, \n
    this function calculates rho and c for normalization \n

    c = \sum_0^(m-1) z_j\n
    rho = \Omega_0^(m-1) |c-z_j|^1/m
    
    :param Z: set of Leja points

    '''
    m = len(Z)
    c = sum(Z) / m
    rho = 1
    for i in range(m):
        rho *= np.abs(c - Z[i])**(1/m)
    return rho,c



def RestartedNewton(v,A,dt,m):
    '''
    Docstring for RestartedNewton
    
    :param v: input vector v \in C^N
    :param A: operator A \in C^(N times N)
    :param dt: time step 
    :param m: maximum size of Hessenberg matrices
    '''
    mat_size = len(A)
    A_0 = np.complex128(np.zeros([mat_size,mat_size]))
    Z_0 = np.zeros(mat_size)
    omega_0 = np.complex128(np.zeros([mat_size,1]))
    v_0 = copy.deepcopy(v)
    beta = np.linalg.norm(v_0,2)
    v_0 /= beta
    s = 0
    converged = 0
    while not converged:
        U,H,Z,m = arnoldi.Arnoldi(A,dt,v_0,m)
        if m == 0 and s == 0:
            return np.exp(-1j * beta * H[1][1]) * v_0
        rho,c = normalizaiton(Z)
        Z_p = ExtendLeja()
        if 0: converged = 1
    return 0



def max_dist(seq:list):
    dist = [1] * (len(seq) - 1)
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i != j:
                dist[i] *= np.abs(seq[i] - seq[j])
    max_index = dist.find(np.max(dist))
    return seq[max_index]

def lejaRadius(Z):
    r_max = max(np.abs(Z))
    
    return 1.2 * r_max

def ExtendLeja(seq_existing:np.ndarray,seq_candidate:list,n_choose:int):
    '''
    
    Docstring for ExtendLeja
    
    :param seq_existing: n existing Leha points
    :param seq_candidate: new candidate points (Rizt valutes)
    :param n_choose: number n_choose of points to pick from seq_candidate

    '''
    n_0 = len(seq_existing)
    seq_new = np.zeros(n_0 + n_choose)
    seq_new[:n_0] = seq_existing
    if not n_0:
        z = np.max(np.abs(seq_candidate))
        seq_new[0] = z
        seq_candidate.pop(seq_candidate.find(z))
        n_0 = 1
    for _ in range(n_0,n_choose):
        z_i = max_dist(seq_candidate)
        seq_existing.append(z_i)
        seq_candidate.pop(seq_candidate.find(z_i))
    return seq_existing

def coeffs_mtp(zs,k,n):
    val = 1
    for i in range(n):
        val *= (zs[k]-zs[i])
    return val

def ExtendNewtonCoeffs(A_s,Z_sp,rho,c):
    '''
    Docstring for ExtendNewtonCoeffs
    
    :param A_s: A_s = [a_0...a_(ns-1)] of n_s Newton coefficients from previous iteration
    :param Z_sp: Z_sp = [z_0...z_(ns-1+m)] of Leja points
    :param rho: normalization radius
    :param c: normalization center
    '''
    A_sp = A_s
    n_0 = len(A_s)
    n_s = len(A_s)
    m = len(Z_sp) - n_s
    def mat_exp(z):
        return 1
    if n_s == 0:
        a_0 = mat_exp(Z_sp[0])
        A_sp.append(a_0)
        n_0 = 1
    for k in range(n_0,n_s-1+m):
        a_k = (mat_exp(Z_sp[k])-a_0-np.sum([A_sp[n]*coeffs_mtp(Z_sp,k,n) for n in range(1,k)]))/coeffs_mtp(Z_sp,k,k)
        A_sp.append(a_k)
    return A_sp