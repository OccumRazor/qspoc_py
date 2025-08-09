import numpy as np,copy
from .read_write import matrix2text
from . import Weyl
from scipy.linalg import svd

def tau(state,ref):
    res=0j
    for i in range(len(state)):
        res+=state[i]*np.conjugate(ref[i])
    try:
        return res[0]
    except TypeError:
        return res

def chis_taus(state,ref):
    #return -tau(state,ref) * ref
    return ref

def JT_re(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            tau_val = tau(states[i],refs[i])
            val += 1-np.real(tau_val)
        return val / len(states)
    else:
        tau_val = tau(states,refs)
        return 1-np.real(tau_val)

def J_T_abs(state,ref):
    return 1-np.abs(tau(state,ref))

#def chis_tau(state,ref):
def chis_tau(ref,state):
    return ref

#def JT_tau(states,refs):
def JT_tau(refs,states):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            tau_val = tau(states[i],refs[i])
            val += np.real(tau_val*np.conjugate(tau_val))
        return val / len(states)
    else:
        tau_val = tau(states,refs)
        return np.real(tau_val*np.conjugate(tau_val))

#def chis_ss(states,refs):
def chis_ss(refs,states):
    if isinstance(states,list):
        chis = []
        for i in range(len(states)):
            chis.append(tau(states[i],refs[i]) * refs[i])
        chis_norm = [np.linalg.norm(chi,2) for chi in chis]
        if any([chis_norm[i] == 0 for i in range(len(states))]):
            print(f'zero norm of chi_T detected, norm of chis_T: {chis_norm}')
        return chis
    else:
        return tau(states[i],refs[i]) * refs[i]

#def JT_ss(states,refs):
def JT_ss(refs,states):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            tau_val = tau(states[i],refs[i])
            val += 1-np.real(tau_val*np.conjugate(tau_val))
        return val / len(states)
    else:
        tau_val = tau(states,refs)
        return 1-np.real(tau_val*np.conjugate(tau_val))

def JT_PE(basis,w):
    if w < 0: w = 0
    if w > 1: w = 1
    def JT(psi_T):
        U = Weyl.state2gate(basis,psi_T)
        c1,c2,c3 = Weyl.c1c2c3(Weyl.from_magic(U))
        g1,g2,g3 = Weyl.c2g(c1,c2,c3)
        conc = Weyl.concurrence(c1,c2,c3)
        Delta_U = 1 - np.real(np.trace(np.matmul(np.conjugate(np.transpose(U)),U))) / 4
        F_PE = (1-w) * (g3 * np.sqrt(g1 ** 2 + g2 ** 2) - g1 + 0.0) + w * Delta_U
        return [F_PE,conc,w*Delta_U]
    def U_info(psi_T):
        U = Weyl.state2gate(basis,psi_T)
        U_cano = Weyl.from_magic(U)
        c1,c2,c3 = Weyl.c1c2c3(U_cano)
        U_text = f'c1: {c1} c2: {c2} c3: {c3}\n'
        U_text += matrix2text(U_cano)
        Delta_U = 1 - np.real(np.trace(np.matmul(np.conjugate(np.transpose(U_cano)),U_cano))) / 4
        U_text += f'\n Norm of U: {Delta_U}\n'
        U_svd,s,Vh = svd(U_cano)
        U_cano = np.matmul(U_svd,Vh)
        c1,c2,c3 = Weyl.c1c2c3(U_cano)
        U_text += f'Above is before svd reconstruction\nc1: {c1} c2: {c2} c3: {c3}\n'
        U_text += matrix2text(U_cano)
        Delta_U = 1 - np.real(np.trace(np.matmul(np.conjugate(np.transpose(U_cano)),U_cano))) / 4
        U_text += f'\n Norm of U: {Delta_U}\n'
        return U_text
    return JT,U_info

def chis_PE(canonical_basis,w):
    if w < 0: w = 0
    if w > 1: w = 1
    bell_basis = Weyl.get_bell_basis(canonical_basis)
    bell_basis_TP = copy.deepcopy(bell_basis)
    state_size = bell_basis_TP[0].shape[0]
    for i in range(4):
        bell_basis_TP[i] = np.conjugate(np.reshape(bell_basis_TP[i],state_size))
    def chi_constructor(psi_T):
        psi_T_TP = copy.deepcopy(psi_T)
        for i in range(4):
            psi_T_TP[i] = np.reshape(psi_T_TP[i],state_size)
        UB = Weyl.state2gate(bell_basis, psi_T)
        A = (Weyl.Qmagic * Weyl._get_a_kl_PE(UB)) / 2
        chis = Weyl.mapped_basis(A, canonical_basis)
        # unitarity corrections
        n = len(psi_T)
        chis_out = []
        for i in range(n):
            bell_proj = np.zeros(shape=psi_T[i].shape,dtype = np.complex128)
            for j in range(n):
                bell_proj += np.inner(bell_basis_TP[j],psi_T_TP[i]) \
                             * bell_basis[j]
            chis_out.append((1.0-w) * chis[i] + 0.25*w * bell_proj)
        return chis_out
    return chi_constructor

