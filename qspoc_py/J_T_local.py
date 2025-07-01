import numpy as np,copy

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

def J_T_re(state,ref):
    return 1-np.real(tau(state,ref))

def J_T_abs(state,ref):
    return 1-np.abs(tau(state,ref))

def chis_tau(state,ref):
    return ref

def JT_tau(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            tau_val = tau(states[i],refs[i])
            val += np.real(tau_val*np.conjugate(tau_val))
        return val / len(states)
    else:
        tau_val = tau(states,refs)
        return np.real(tau_val*np.conjugate(tau_val))

def chis_ss(states,refs):
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

def JT_ss(states,refs):
    if isinstance(states,list):
        val = 0
        for i in range(len(states)):
            tau_val = tau(states[i],refs[i])
            val += 1-np.real(tau_val*np.conjugate(tau_val))
        return val / len(states)
    else:
        tau_val = tau(states,refs)
        return 1-np.real(tau_val*np.conjugate(tau_val))

from . import Weyl

def JT_PE(states,basis):
    U = Weyl.stat2gate(basis,states)
    c1,c2,c3 = Weyl.c1c2c3(Weyl.from_magic(U))
    g1,g2,g3 = Weyl.c2g(c1,c2,c3)
    conc = Weyl.concurrence(c1,c2,c3)
    F_PE = g3 * np.sqrt(g1 ** 2 + g2 ** 2) - g1 + 0.0
    print("    F_PE: %f\n    gate conc.: %f" % (F_PE, conc))
    return F_PE

def stat2gate(basis,states):
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

def chis_PE(canonical_basis,w):
    if w < 0: w = 0
    if w > 1: w = 1
    bell_basis = Weyl.get_bell_basis(canonical_basis)
    bell_basis_TP = copy.deepcopy(bell_basis)
    state_size = bell_basis_TP[0].shape[0]
    for i in range(4):
        bell_basis_TP[i] = np.conjugate(np.reshape(bell_basis_TP[i],state_size))
    def chi_constructor(fw_states_T, **kwargs):
        psi_T_TP = copy.deepcopy(fw_states_T)
        for i in range(4):
            psi_T_TP[i] = np.reshape(psi_T_TP[i],state_size)
        # *args is ignored, it exists so that the chi_constructor fits the
        # krotov API directly
        UB = Weyl.stat2gate(bell_basis, fw_states_T)
        A = (Weyl.Qmagic * Weyl._get_a_kl_PE(UB)) / 2

        chis = Weyl.mapped_basis(A, canonical_basis)

        # unitarity corrections
        n = len(fw_states_T)
        chis_out = []
        for i in range(n):
            bell_proj = np.zeros(shape=fw_states_T[i].shape,dtype = np.complex128)
            for j in range(n):
                bell_proj += np.inner(bell_basis_TP[j],psi_T_TP[i]) \
                             * bell_basis[j]
            chis_out.append((1.0-w) * chis[i] + 0.25*w * bell_proj)
        return chis_out
    return chi_constructor

