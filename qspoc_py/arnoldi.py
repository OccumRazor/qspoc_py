import numpy as np,copy

"""
def Arnoldi(A,dt,v_s,m_max):
    '''
    Docstring for Arnoldi
    
    :param A: operator
    :param dt: time step
    :param v_s: input vector
    :param m_max: maximum order
    '''
    tol = 1e-9
    beta = np.linalg.norm(v_s,2)
    u_i = v_s / beta
    Z = []
    m = m_max
    H = np.zeros([m+1,m+1])
    U = [u_i]
    for i in range(1,m_max+1):
        u_ip = np.matmul(A,u_i)
        for j in range(i):
            H[i][j] = np.inner(np.conjugate(u_i),u_ip) * dt
            u_ip -= H[i][j] / dt *u_i
        H_clip = 1
        Z.append(np.linalg.eigvals(H_clip))
        h_next = np.linalg.norm(u_ip)
        if np.abs(h_next) < tol:
            m = i
            break
        u_i = copy.deepcopy(u_ip)
        u_i /= h_next
        U.append(u_i)
        H[j+1,j] = h_next * dt
    return U,H,Z,m
"""

def Arnoldi(Hess,q,m,psi,H,dt:float,extended=True,tol=1e-15):
    dim_hess = m
    if extended: dim_hess += 1
    assert np.shape(Hess,0) >= dim_hess and np.shape(Hess,1) >= dim_hess
    assert len(q) >= m + 1
    q[:,0] = psi
    for i in range(m):
        q[:,i+1] = np.matmul(H,q[:,i])
        for j in range(i):
            Hess[j,i] = np.inner(q[:,i+1],q[:,j]) * dt
            q[i+1] -= q[j] * Hess[j,i] / dt
        if i < m or extended:
            h = np.norm(q[:,i+1])
            Hess[i+1,i] = dt * h
            if h < tol:
                m = j
                break
            q[i+1] *= 1/h
    return m


def extend_arnoldi(Hess,q,m,H,dt,tol = 1e-15):
    h = np.linalg.norm(q[m],2)
    if h < tol: return m
    Hess[m-1,m-2] = dt * h
    q[m-1] *= 1/h
    q[m] = np.matmul(H,q[m-1])
    for i in range(m):
        Hess[i,m] = dt * np.inner(np.conj(q[i]),q[m])
        q[m] -= Hess[i][m] / dt * q[i]
    assert all([Hess[m-1][i] == 0 for i in range(len(Hess[m-1]))])
    return Hess

def diagonalize_hessenberg_matrix(Hess,m,accumulate = False):
    '''
    Docstring for diagonalize_hessenberg_matrix

    Diagonalize the m x m top left sub-matrix of a given Hessenberg matrix

    '''

    j_min = m
    j_max = m
    if accumulate:
        j_min = 0
        eigenvals = np.zeros(0.5*m*(m+1),dtype = np.complex128)
    else:
        eigenvals = np.zeros(m,dtype = np.complex128)
    
    offset = 0
    for j in range(j_min,j_max+1):
        if j == 1:
            eigenvals[0] = Hess[0,0]
        elif j == 2:
            a = Hess[0,0]
            c = Hess[1,0]
            b = Hess[0,1]
            d = Hess[1,1]
            s = np.sqrt(a**2+4*b*c-2*a*d+d**2)
            eigenvals[offset] = 0.5*(a+d-s)
            eigenvals[offset+1] = 0.5*(a+d+s)
        else:
            eigenvals[offset:offset+j] = np.linalg.eigvals(Hess[:j,:j])
        offset += j

    return eigenvals