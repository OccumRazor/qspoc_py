import numpy as np,copy

Qmagic = (1.0 / np.sqrt(2.0)) * np.array(
        [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
        dtype=np.complex128)
Qmagic_HC = np.conjugate(np.transpose(Qmagic))

SySy = np.array(
        [[+0, 0, 0, -1], [+0, 0, 1, 0], [+0, 1, 0, 0], [-1, 0, 0, 0]],
        dtype=np.complex128)

def from_magic(A):
    """ The inverse of :func:`.to_magic`"""
    if A.shape != (4, 4):
        raise ValueError("Gates must have a 4×4 shape")
    return np.matmul(Qmagic,np.matmul(A,Qmagic_HC))

def state2gate(basis,states):
    basis = copy.deepcopy(basis)
    states = copy.deepcopy(states)
    state_size = basis[0].shape[0]
    for i in range(4):
        basis[i] = np.conjugate(np.reshape(basis[i],state_size))
        states[i] = np.conjugate(np.reshape(states[i],state_size))
    U = np.zeros([4,4],dtype=np.complex128)
    for j in range(4):
        for i in range(4):
            U[i,j] = np.inner(basis[i],states[j])
    return U

def c1c2c3(U):
    U_tilde = np.matmul(SySy,np.matmul(np.transpose(U),SySy))
    ev = np.linalg.eigvals(np.matmul(U, U_tilde)/np.sqrt(complex(np.linalg.det(U))))
    two_S = np.angle(ev) / np.pi
    for i in range(len(two_S)):
        if two_S[i] <= -0.5:
            two_S[i] += 2.0
    S = np.sort(two_S / 2.0)[::-1]  # sort decreasing
    n = int(round(sum(S)))
    S -= np.r_[np.ones(n), np.zeros(4-n)]
    S = np.roll(S, -n)
    M = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    c1, c2, c3 = np.dot(M, S[:3])
    if c3 < 0:
        c1 = 1 - c1
        c3 = -c3
    return c1+0.0,c2+0.0,c3+0.0


def c2g(c1,c2,c3):
    c1 *= np.pi
    c2 *= np.pi
    c3 *= np.pi
    g1 = np.cos(c1) ** 2 * np.cos(c2) ** 2 * np.cos(c3) ** 2- np.sin(c1) ** 2 * np.sin(c2) ** 2 * np.sin(c3) ** 2+ 0.0
    g2 = 0.25 * np.sin(2 * c1) * np.sin(2 * c2) * np.sin(2 * c3) + 0.0
    g3 = 4 * g1 - np.cos(2 * c1) * np.cos(2 * c2) * np.cos(2 * c3) + 0.0
    return g1, g2, g3


def concurrence(c1,c2,c3):
    if ((c1 + c2) >= 0.5) and (c1 - c2 <= 0.5) and ((c2 + c3) <= 0.5):
        # if we're inside the perfect-entangler polyhedron in the Weyl chamber
        # the concurrence is 1 by definition. the "regular" formula gives wrong
        # results in this case.
        return 1
    else:
        c1_c2_c3 = np.array([c1, c2, c3])
        c3_c1_c2 = np.roll(c1_c2_c3, 1)
        m = np.concatenate((c1_c2_c3 - c3_c1_c2, c1_c2_c3 + c3_c1_c2))
        res = np.max(abs(np.sin(np.pi * m)))
        if abs(res) < 1e-15:
            return 0
        elif abs(res - 1) < 1e-15:
            return 1
        return res
    
def get_bell_basis(canonical_basis):
    return mapped_basis(Qmagic, canonical_basis)

def mapped_basis(gate, basis):
    return [sum([complex(gate[i, j]) * basis[i] for i in range(gate.shape[0])])
            for j in range(gate.shape[1])]


def _get_a_kl_PE(UB):
    """Return the 4×4 `A_kl` coefficient matrix (:class:`qutip.Qobj`)
    for the perfect-entanglers functional, for a given gate `UB` in the Bell
    basis.
    """

    # Compute auxiliary scalar quantities
    detU = _cmat4_det(UB)
    rcdetU1 = 1.0 / detU
    #mU = UB.trans() * UB
    mU = np.matmul(np.transpose(UB),UB)
    #tr_mU = mU.tr()
    tr_mU = np.trace(mU)
    #tr_mU_sq = (mU*mU).tr()
    tr_mU_sq = np.trace(np.matmul(mU,mU))
    g3 = 0.25 * (tr_mU**2 - tr_mU_sq)
    Re2Tr = np.real(tr_mU)**2 / 16.0 # This contains already the 1/16 factor
    Im2Tr = np.imag(tr_mU)**2 / 16.0 # This contains already the 1/16 factor
    Tr2 = tr_mU**2 / 16.0            # This contains already the 1/16 factor

    # `alpha` and `beta` are the real and imaginary part of `UB`, respectively
    alpha = np.zeros(shape=(4,4))
    beta  = np.zeros(shape=(4,4))
    for i in range(4):
        for j in range(4):
            alpha[i,j] = np.real(UB[i,j])
            beta[i,j]  = np.imag(UB[i,j])

    # The c1c2c3 function expects `UB` to be given in the canonical basis, thus
    # we have to convert it from the Bell basis on input
    c1, c2, c3 = c1c2c3(from_magic(UB))

    drcdetU1dAlpha = np.zeros(shape=(4,4), dtype=complex)
    drcdetU1dBeta  = np.zeros(shape=(4,4), dtype=complex)
    dg3dAlpha      = np.zeros(shape=(4,4), dtype=complex)
    dg3dBeta       = np.zeros(shape=(4,4), dtype=complex)
    dRe2TrdAlpha   = np.zeros(shape=(4,4))
    dRe2TrdBeta    = np.zeros(shape=(4,4))
    dIm2TrdAlpha   = np.zeros(shape=(4,4))
    dIm2TrdBeta    = np.zeros(shape=(4,4))
    dTr2dAlpha     = np.zeros(shape=(4,4), dtype=complex)
    dTr2dBeta      = np.zeros(shape=(4,4), dtype=complex)

    # Compute auxiliary matrix quantities
    for a in range(4):
        for b in range(4):

            drcdetU1dAlpha[a,b] = - _dDetUdAlpha(UB,a,b) / detU**2
            drcdetU1dBeta[a,b]  = - _dDetUdBeta(UB,a,b)  / detU**2

            for k in range(4):
                for i in range(4):
                    dg3dAlpha[a,b] = dg3dAlpha[a,b]                           \
                                 +       alpha[a,b] * alpha[k,i] * alpha[k,i] \
                                 -       alpha[a,b] * beta[k,i]  * beta[k,i]  \
                                 - 2.0 * beta[a,b]  * alpha[k,i] * beta[k,i]  \
                                 -       alpha[k,i] * alpha[a,i] * alpha[k,b] \
                                 +       alpha[k,b] * beta[a,i]  * beta[k,i]  \
                                 + 2.0 * beta[k,b]  * alpha[a,i] * beta[k,i]
                    dg3dBeta[a,b]  = dg3dBeta[a,b]                            \
                                 +       beta[a,b]  * beta[k,i]  * beta[k,i]  \
                                 -       beta[a,b]  * alpha[k,i] * alpha[k,i] \
                                 - 2.0 * alpha[a,b] * alpha[k,i] * beta[k,i]  \
                                 -       beta[k,i]  * beta[a,i]  * beta[k,b]  \
                                 +       beta[k,b]  * alpha[a,i] * alpha[k,i] \
                                 + 2.0 * alpha[k,b] * beta[a,i]  * alpha[k,i]
                    dg3dAlpha[a,b] = dg3dAlpha[a,b] + 1j * (                  \
                                 +       beta[a,b]  * alpha[k,i] * alpha[k,i] \
                                 -       beta[a,b]  * beta[k,i]  * beta[k,i]  \
                                 + 2.0 * alpha[a,b] * alpha[k,i] * beta[k,i]  \
                                 -       alpha[a,i] * alpha[k,i] * beta[k,b]  \
                                 -       alpha[k,b] * alpha[k,i] * beta[a,i]  \
                                 -       alpha[k,b] * alpha[a,i] * beta[k,i]  \
                                 +       beta[a,i]  * beta[k,i]  * beta[k,b]  \
                                 )
                    dg3dBeta[a,b]  = dg3dBeta[a,b] + 1j * (                   \
                                 -       alpha[a,b] * beta[k,i]  * beta[k,i]  \
                                 +       alpha[a,b] * alpha[k,i] * alpha[k,i] \
                                 - 2.0 * beta[a,b]  * beta[k,i]  * alpha[k,i] \
                                 +       beta[a,i]  * beta[k,i]  * alpha[k,b] \
                                 +       beta[k,b]  * beta[k,i]  * alpha[a,i] \
                                 +       beta[k,b]  * beta[a,i]  * alpha[k,i] \
                                 -       alpha[a,i] * alpha[k,i] * alpha[k,b] \
                                 )

            for k in range(4):
                for i in range(4):
                    dTr2dAlpha[a,b] = dTr2dAlpha[a,b]                         \
                                + 0.25 * alpha[a,b] * alpha[k,i] * alpha[k,i] \
                                - 0.25 * alpha[a,b] * beta[k,i]  * beta[k,i]  \
                                - 0.50 * beta[a,b]  * alpha[k,i] * beta[k,i]
                    dTr2dBeta[a,b]  = dTr2dBeta[a,b]                          \
                                + 0.25 * beta[a,b]  * beta[k,i]  * beta[k,i]  \
                                - 0.25 * beta[a,b]  * alpha[k,i] * alpha[k,i] \
                                - 0.50 * alpha[a,b] * beta[k,i]  * alpha[k,i]
                    dTr2dAlpha[a,b] = dTr2dAlpha[a,b] + 1j * (                \
                                + 0.25 * beta[a,b]  * alpha[k,i] * alpha[k,i] \
                                - 0.25 * beta[a,b]  * beta[k,i]  * beta[k,i]  \
                                + 0.50 * alpha[a,b] * alpha[k,i] * beta[k,i]  \
                                )
                    dTr2dBeta[a,b]  = dTr2dBeta[a,b] + 1j * (                 \
                                - 0.25 * alpha[a,b] * beta[k,i]  * beta[k,i]  \
                                + 0.25 * alpha[a,b] * alpha[k,i] * alpha[k,i] \
                                - 0.50 * beta[a,b]  * beta[k,i]  * alpha[k,i] \
                                )

            for k in range(4):
                for i in range(4):
                    dRe2TrdAlpha[a,b] = dRe2TrdAlpha[a,b] + np.real(          \
                                  0.25 * alpha[a,b] * alpha[k,i] * alpha[k,i] \
                                - 0.25 * alpha[a,b] * beta[k,i] * beta[k,i]   \
                                )
                    dRe2TrdBeta[a,b]  = dRe2TrdBeta[a,b] + np.real(           \
                                   0.25 * beta[a,b] * beta[k,i] * beta[k,i]   \
                                 - 0.25 * beta[a,b] * alpha[k,i] * alpha[k,i] \
                                 )
                    dIm2TrdAlpha[a,b] = dIm2TrdAlpha[a,b] + np.real(          \
                                    0.50 * alpha[k,i] * beta[a,b] * beta[k,i] \
                                    )
                    dIm2TrdBeta[a,b]  = dIm2TrdBeta[a,b] + np.real(           \
                                   0.50 * beta[k,i] * alpha[a,b] * alpha[k,i] \
                                   )

    # Construct the a_kl coefficients with the previously computed quantities
    a_kl_coeffs = np.zeros(shape=(4,4), dtype=complex)
    for l in range(4):
        for k in range(4):
            a_kl_coeffs[k,l] =                                                \
            np.real(                                                          \
              - drcdetU1dAlpha[k,l] * g3 * Re2Tr                              \
              - rcdetU1 * dg3dAlpha[k,l] * Re2Tr                              \
              - rcdetU1 * g3 * dRe2TrdAlpha[k,l]                              \
              - drcdetU1dAlpha[k,l] * g3 * Im2Tr                              \
              - rcdetU1 * dg3dAlpha[k,l] * Im2Tr                              \
              - rcdetU1 * g3 * dIm2TrdAlpha[k,l]                              \
              + drcdetU1dAlpha[k,l] * Tr2                                     \
              + rcdetU1 * dTr2dAlpha[k,l])                                    \
            + 1j *                                                            \
            np.real(                                                          \
              - drcdetU1dBeta[k,l] * g3 * Re2Tr                               \
              - rcdetU1 * dg3dBeta[k,l] * Re2Tr                               \
              - rcdetU1 * g3 * dRe2TrdBeta[k,l]                               \
              - drcdetU1dBeta[k,l] * g3 * Im2Tr                               \
              - rcdetU1 * dg3dBeta[k,l] * Im2Tr                               \
              - rcdetU1 * g3 * dIm2TrdBeta[k,l]                               \
              + drcdetU1dBeta[k,l] * Tr2                                      \
              + rcdetU1 * dTr2dBeta[k,l])

    if (c2 + c3 > 0.5):
        # UB is in the W1 region of the Weyl chamber (between the PE polyhedron
        # and the SWAP gate at the A3 point). In this region, the PE-functional
        # has the wrong sign (cf. Fig 6.1 in Goerz, PhD Thesis, Kassel).
        a_kl_coeffs = -a_kl_coeffs
        # Without this corrections, gates would be pushed towards [SWAP]

    return a_kl_coeffs


def _cmat4_det(m):
    """Calculates the complex determinant of a 4x4 matrix.

    Args:
        m (qutip.Qobj): A 4x4 quantum gate for a two-qubit system

    Returns:
        float: The determinant $\det(m)$
    """
    d =   m[0,3]*m[1,2]*m[2,1]*m[3,0] - m[0,2]*m[1,3]*m[2,1]*m[3,0]           \
        - m[0,3]*m[1,1]*m[2,2]*m[3,0] + m[0,1]*m[1,3]*m[2,2]*m[3,0]           \
        + m[0,2]*m[1,1]*m[2,3]*m[3,0] - m[0,1]*m[1,2]*m[2,3]*m[3,0]           \
        - m[0,3]*m[1,2]*m[2,0]*m[3,1] + m[0,2]*m[1,3]*m[2,0]*m[3,1]           \
        + m[0,3]*m[1,0]*m[2,2]*m[3,1] - m[0,0]*m[1,3]*m[2,2]*m[3,1]           \
        - m[0,2]*m[1,0]*m[2,3]*m[3,1] + m[0,0]*m[1,2]*m[2,3]*m[3,1]           \
        + m[0,3]*m[1,1]*m[2,0]*m[3,2] - m[0,1]*m[1,3]*m[2,0]*m[3,2]           \
        - m[0,3]*m[1,0]*m[2,1]*m[3,2] + m[0,0]*m[1,3]*m[2,1]*m[3,2]           \
        + m[0,1]*m[1,0]*m[2,3]*m[3,2] - m[0,0]*m[1,1]*m[2,3]*m[3,2]           \
        - m[0,2]*m[1,1]*m[2,0]*m[3,3] + m[0,1]*m[1,2]*m[2,0]*m[3,3]           \
        + m[0,2]*m[1,0]*m[2,1]*m[3,3] - m[0,0]*m[1,2]*m[2,1]*m[3,3]           \
        - m[0,1]*m[1,0]*m[2,2]*m[3,3] + m[0,0]*m[1,1]*m[2,2]*m[3,3]
    return d


def _dDetUdAlpha(U,a,b):
    """Calculates

    .. math::

        \partial \det(U) / \partial \\alpha_{a,b} = \det(U'_{a,b})

    where $U'$ is constructed from $U$ by replacing the row $a$ with the unit
    vector $b$ and $\\alpha$ is the real part of $U$.

    Args:
        U (qutip.Qobj): A 4x4 quantum gate for a two-qubit system
        a (integer): An integer specifying the derivative
        b (integer): An integer specifying the derivative

    Returns:
        float: The derivative as specified by $a$ and $b$
    """
    U_prime = np.zeros(shape=(4,4), dtype=complex)
    for i in range(4):
        for j in range(4):
            if i==a:
                U_prime[i,j] = 0
            else:
                U_prime[i,j] = U[i,j]
    U_prime[a,b] = 1
    return _cmat4_det(U_prime)


def _dDetUdBeta(U,a,b):
    """Calculates

    .. math::

        \partial \det(U) / \partial \\beta_{a,b} = \det(U'_{a,b})

    where $U'$ is constructed from $U$ by replacing the row $a$ with the unit
    vector $b$ and $\\beta$ is the imaginary part of $U$.

    Args:
        U (qutip.Qobj): A 4x4 quantum gate for a two-qubit system
        a (integer): An integer specifying the derivative
        b (integer): An integer specifying the derivative

    Returns:
        float: The derivative as specified by $a$ and $b$
    """
    return 1j*_dDetUdAlpha(U,a,b)