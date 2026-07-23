import numpy as np,copy
from scipy.linalg import expm
from . import arnoldi

class NewtonWrk:
    '''
    v: state vector
    arnoldi_vecs: Array
    a: OffsetVector Newton Coefficients 
    leja_points: OffsetVector
    '''
    def __init__(self,m_max:int,dim:int):
        self.m_max = m_max
        self.v = np.zeros(dim,dtype=np.complex128)
        self.arnoldi_vecs = np.zeros([dim,m_max + 1],dtype = np.complex128)
        self.a = np.zeros(0,dtype=np.complex128)
        self.leja_points = np.zeros(0,dtype=np.complex128)
        self.radius = 0.
        self.n_leja = 0
        self.n_a = 0
        self.restarts = 200

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

def lejaRadius(Z):
    r_max = max(np.abs(Z))
    return 1.2 * r_max

def ExtendLeja(wrk:NewtonWrk,seq_candidate:np.ndarray,n_choose:int):
    '''
    
    Docstring for ExtendLeja
    
    :param seq_existing: n existing Leha points
    :param seq_candidate: new candidate points (Rizt valutes)
    :param n_choose: number n_choose of points to pick from seq_candidate

    '''
    seq_existing = wrk.leja_points
    assert n_choose <= np.size(seq_candidate)
    n = np.size(seq_existing)
    n_last = n + n_choose
    seq_new = np.zeros(n_last,dtype=np.complex128)
    seq_new[:n] = seq_existing
    n_0 = 0
    if not n:
        z = np.max(np.abs(seq_candidate))
        z_loc = np.where(np.abs(seq_candidate) == z)[0][0]
        seq_new[0] = seq_candidate[z_loc]
        seq_candidate = np.delete(seq_candidate,z_loc)
        n_0 = 1
    for i_new in range(n_0,n_choose):
        p_max = 0
        i_max = 0
        for i in range(len(seq_candidate)):
            p = 1
            for j in range(n_0 + i_new):
                delta = np.abs(seq_candidate[i] - seq_new[j])
                p *= delta ** (1/n_last)
            if p > p_max:
                p_max = p
                i_max = i
        seq_new[n + i_new] = seq_candidate[i_max]
        seq_candidate[i_max] = seq_candidate[-1-i_new]
    wrk.leja_points = seq_new
    return n_last

def coeffs_mtp(zs,k,n):
    val = 1
    for i in range(n):
        val *= (zs[k]-zs[i])
    return val

def ExtendNewtonCoeffs(wrk:NewtonWrk,n_leja:int,radius,func):
    '''
    Docstring for ExtendNewtonCoeffs
    
    :param coeffs: coeffs = [c_0...c_(ns-1)] of n_s Newton coefficients from previous iteration
    :param leja_points: leja_points = [l_0...l_(ns-1+m)] of Leja points
    :param n_leja: choose n_leja from leja_points
    :param leja_points: normalization radius
    '''
    coeffs = wrk.a
    leja_points = wrk.leja_points
    n0 = len(coeffs)
    #coeffs = np.resize(coeffs,n0 + n_leja)
    #coeffs[n0:] = np.zeros(n_leja,dtype=np.complex128)
    coeffs = np.resize(coeffs,n_leja)
    coeffs[n0:] = np.zeros(n_leja - n0,dtype=np.complex128)
    assert radius > 0
    if n0 == 0:
        coeffs[0] = func(leja_points[0])
        n0 = 1
    #print(leja_points)
    for k in range(n0,n_leja):
        d = 1 + 0j
        pn = 0j
        #print(f'k:{k}')
        for n in range(k):
            zd = leja_points[k] - leja_points[n]
            #print(f'n:{n},zd:{zd}')
            d *= zd / radius
            pn += coeffs[n] * d
        zd = leja_points[k] - leja_points[k-1]
        d *= zd / radius
        assert np.abs(d) > 1e-200
        coeffs[k] = (func(leja_points[k]) - coeffs[0] - pn) / d
    wrk.a = coeffs
    return len(coeffs)

def Newton(psi,H,dt,func):
    tol = 1e-10
    max_restarts = 50
    dim = np.size(psi)
    wrk = NewtonWrk(10,dim)
    m = wrk.m_max
    R = np.zeros(wrk.m_max+1,dtype=np.complex128)
    P = np.zeros(wrk.m_max+1,dtype=np.complex128)
    R_abs = np.zeros(wrk.m_max+1,dtype=np.float64)
    Hess = np.zeros((wrk.m_max+1,wrk.m_max+1),dtype=np.complex128)
    n_a = 0
    n_leja = 0
    assert dt != 0
    wrk.v = psi
    s = 0
    beta = np.linalg.norm(wrk.v,2)
    wrk.v /= beta
    while True:
        print(f'iteration: {s}')
        m = arnoldi.Arnoldi(Hess,wrk.arnoldi_vecs,m,wrk.v,H,dt,True)
        if m == 1 and s == 0:
            L = beta * Hess[0,0]
            psi *= func(L)
            break
        ritz = arnoldi.diagonalize_hessenberg_matrix(Hess,m,True)
        if s == 0:
            wrk.radius = lejaRadius(ritz)
        n_s = n_leja
        n_leja = ExtendLeja(wrk,ritz,m)
        n_a = ExtendNewtonCoeffs(wrk,n_leja,wrk.radius,func)
        assert n_a == n_leja
        if len(R) != m+1:
            np.resize(R,m+1)
            np.resize(P,m+1)
            np.resize(R_abs,m+1)
        P *= 0
        R *= 0
        R[0] = beta
        P[0] = wrk.a[n_s] * beta
        for k in range(1,m):
            z = wrk.leja_points[n_s+k-1]
            R = (np.matmul(Hess,R) - z * R) / wrk.radius
            P += wrk.a[n_s+k] * R
        if s == 0:
            psi *= 0
        #print(wrk.arnoldi_vecs)
        #print(P)
        for i in range(m):
            #print(P[i])
            #print(wrk.arnoldi_vecs[:,i])
            #print(psi)
            psi += P[i] * np.reshape(wrk.arnoldi_vecs[:,i],(dim,1))
        print(psi)
        print(np.linalg.norm(psi,2))
        R = (np.matmul(Hess,R) - wrk.leja_points[n_s+m-1] * R) / wrk.radius
        R_abs = np.abs(R)
        beta = np.linalg.norm(R_abs)
        R /= beta
        #print(wrk.arnoldi_vecs[:,0])
        #print(wrk.arnoldi_vecs[:,0].T)
        #print(wrk.v)
        wrk.arnoldi_vecs[:,0] = wrk.v.T[0]
        wrk.v *= R[1]
        for i in range(1,m+1):
            wrk.v += R[i] * np.reshape(wrk.arnoldi_vecs[:,i],(dim,1))
        
        psi_relerr = beta * np.abs(wrk.a[n_a-1]) / (1+np.linalg.norm(psi,2))
        if psi_relerr < tol:
            break
        else:
            s += 1
            assert s <= max_restarts
        
    wrk.restarts = s
    wrk.n_leja = n_leja
    wrk.n_a = n_a
    return psi

