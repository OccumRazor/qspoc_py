import numpy as np,copy
from scipy.linalg import expm
#from . import arnoldi

def diagonalize_hessenberg_matrix(Hess,m,accumulate = False):
    '''
    Docstring for diagonalize_hessenberg_matrix

    Diagonalize the m x m top left sub-matrix of a given Hessenberg matrix

    '''
    j_min = m - 1
    j_max = m
    if accumulate:
        j_min = 0
        eigenvals = np.zeros(int(0.5*m*(m+1)),dtype = np.complex128)
    else:
        eigenvals = np.zeros(m,dtype = np.complex128)
    
    offset = 0
    for j in range(j_min,j_max):
        if j == 0:
            eigenvals[0] = Hess[0,0]
        elif j == 1:
            a = Hess[0,0]
            c = Hess[1,0]
            b = Hess[0,1]
            d = Hess[1,1]
            s = np.sqrt(a**2+4*b*c-2*a*d+d**2)
            eigenvals[offset] = 0.5*(a+d-s)
            eigenvals[offset+1] = 0.5*(a+d+s)
        else:
            eigenvals[offset:offset+j+1] = np.linalg.eigvals(Hess[:j+1,:j+1])
        offset += j + 1
    #print(Hess)
    #print(eigenvals)
    return np.sort(eigenvals)

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

def lejaRadius(Z):
    r_max = max(np.abs(Z))
    return 1.2 * r_max

def coeffs_mtp(zs,k,n):
    val = 1
    for i in range(n):
        val *= (zs[k]-zs[i])
    return val

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
        self.arnoldi_vecs = np.zeros([dim,m_max+1],dtype = np.complex128)
        self.Hess = np.zeros([m_max+1,m_max+1],dtype = np.complex128)
        self.a = np.zeros(0,dtype=np.complex128)
        self.leja_points = np.zeros(0,dtype=np.complex128)
        self.radius = 0.
        self.n_leja = 0
        self.n_a = 0
        self.restarts = 200

    def Arnoldi(self,m,H,dt:float,extended=True,tol=1e-15):
        dim_hess = m
        if extended: dim_hess += 1
        self.Hess *= 0
        self.arnoldi_vecs[:,0] = self.v.T
        for i in range(m):
            self.arnoldi_vecs[:,i+1] = np.matmul(H,self.arnoldi_vecs[:,i])
            for j in range(i+1):
                self.Hess[j,i] = np.inner(np.conj(self.arnoldi_vecs[:,j]),self.arnoldi_vecs[:,i+1]) * dt
                self.arnoldi_vecs[:,i+1] -= self.arnoldi_vecs[:,j] * self.Hess[j,i] / dt
            if i < (m-1) or extended:
                h = np.linalg.norm(self.arnoldi_vecs[:,i+1],2)
                self.Hess[i+1,i] = dt * h
                if h < tol:
                    m = j + 1
                    break
                self.arnoldi_vecs[:,i+1] *= 1/h
        return m

    def extend_arnoldi(self,m,H,dt,tol = 1e-15):
        h = np.linalg.norm(self.arnoldi_vecs[m],2)
        if h < tol: return m
        self.Hess[m-1,m-2] = dt * h
        self.arnoldi_vecs[:,m-1] *= 1/h
        self.arnoldi_vecs[:,m] = np.matmul(H,self.arnoldi_vecs[:,m-1])
        for i in range(m):
            self.Hess[i,m-1] = dt * np.inner(self.arnoldi_vecs[:,i],np.conj(self.arnoldi_vecs[:,m]))
            self.arnoldi_vecs[:,m] -= self.Hess[i][m-1] / dt * self.arnoldi_vecs[:,i]
        assert all([self.Hess[m-1][i] == 0 for i in range(len(self.Hess[m-1]))])
        return self.Hess

    def ExtendLeja(self,seq_candidate:np.ndarray,n_choose:int):
        '''
        
        Docstring for ExtendLeja
        
        :param seq_existing: n existing Leha points
        :param seq_candidate: new candidate points (Rizt valutes)
        :param n_choose: number n_choose of points to pick from seq_candidate

        '''
        seq_existing = self.leja_points
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
            for i in range(len(seq_candidate) - i_new):
                p = 1
                for j in range(n + i_new):
                    delta = np.abs(seq_candidate[i] - seq_new[j])
                    p *= delta ** (1/n_last)
                if p > p_max:
                    p_max = p
                    i_max = i
            seq_new[n + i_new] = seq_candidate[i_max]
            seq_candidate[i_max] = seq_candidate[-1-i_new]
        self.leja_points = seq_new
        print(f'leja points: {self.leja_points}')
        return n_last

    def ExtendNewtonCoeffs(self,n_leja:int,func):
        '''
        Docstring for ExtendNewtonCoeffs
        
        :param coeffs: coeffs = [c_0...c_(ns-1)] of n_s Newton coefficients from previous iteration
        :param leja_points: leja_points = [l_0...l_(ns-1+m)] of Leja points
        :param n_leja: choose n_leja from leja_points
        :param leja_points: normalization radius
        '''
        n0 = len(self.a)
        self.a = np.resize(self.a,n_leja)
        self.a[n0:] = np.zeros(n_leja - n0,dtype=np.complex128)
        assert self.radius > 0
        if n0 == 0:
            self.a[0] = func(self.leja_points[0])
            n0 = 1
        for k in range(n0,n_leja):
            d = 1 + 0j
            pn = 0j
            for n in range(k - 1):
                zd = self.leja_points[k] - self.leja_points[n]
                d *= zd / self.radius
                pn += self.a[n] * d
            zd = self.leja_points[k] - self.leja_points[k-1]
            d *= zd / self.radius
            assert np.abs(d) > 1e-200
            self.a[k] = (func(self.leja_points[k]) - self.a[0] - pn) / d
        print(f'Newton coeffs: {self.a}')
        return len(self.a)


def Newton(psi,H,dt,func):
    tol = 1e-10
    max_restarts = 50
    dim = np.size(psi)
    wrk = NewtonWrk(10,dim)
    m = wrk.m_max
    R = np.zeros(m+1,dtype=np.complex128)
    P = np.zeros(m+1,dtype=np.complex128)
    R_abs = np.zeros(m+1,dtype=np.float64)
    n_a = 0
    n_leja = 0
    assert dt != 0
    wrk.v = psi
    s = 0
    beta = np.linalg.norm(wrk.v,2)
    wrk.v /= beta
    while True:
        print(f'iteration: {s}')
        m = wrk.Arnoldi(m,H,dt,True)
        if m == 1 and s == 0:
            L = beta * wrk.Hess[0,0]
            psi *= func(L)
            break
        ritz = diagonalize_hessenberg_matrix(wrk.Hess,m + 1,True)
        print(f"ritz: {ritz}")
        if s == 0:
            wrk.radius = lejaRadius(ritz)
        print(f'radius: {wrk.radius}')
        n_s = n_leja
        n_leja = wrk.ExtendLeja(ritz,m + 1)
        n_a = wrk.ExtendNewtonCoeffs(n_leja,func)
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
            R = (np.matmul(wrk.Hess,R) - z * R) / wrk.radius
            P += wrk.a[n_s+k] * R
        if s == 0:
            psi *= 0
        for i in range(m):
            psi += P[i] * np.reshape(wrk.arnoldi_vecs[:,i],(dim,1))
        R = (np.matmul(wrk.Hess,R) - wrk.leja_points[n_s+m-1] * R) / wrk.radius
        R_abs = np.abs(R)
        beta = np.linalg.norm(R_abs)
        R /= beta
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

