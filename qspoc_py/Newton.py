import numpy as np

def RestartedNewton(v,A,dt,m):
    mat_size = len(A)
    A_0 = np.complex128(np.zeros([mat_size,mat_size]))
    Z_0 = np.zeros(mat_size)
    convergence = 0
    while not convergence:
        1
    return 0

def Arnoldi(A,dt,v_s,m):
    tol = 1e-9
    return 0

def max_dist(seq:list):
    dist = [1] * (len(seq) - 1)
    for i in range(len(seq)):
        for j in range(len(seq)):
            if i != j:
                dist[i] *= np.abs(seq[i] - seq[j])
    max_index = dist.find(np.max(dist))
    return seq[max_index]

def ExtendLeja(seq_existing:list,seq_candidate:list,n_choose:int):
    n_0 = 1
    if not len(seq_existing):
        z = np.max(np.abs(seq_candidate))
        seq_existing.append(z)
        seq_candidate.pop(seq_candidate.find(z))
        n_0 = 2
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