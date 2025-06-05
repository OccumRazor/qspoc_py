import numpy as np

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


'''
def localFidelity(state,ref='None'):
    if isinstance(state,qutip.Qobj):state=state.full()
    if isinstance(ref,qutip.Qobj):ref=ref.full()
    if localTools.isinstanceVector(state):dm=localTools.densityMatrix(state)
    else:dm=state
    if localTools.isinstanceVector(ref):ref=localTools.densityMatrix(ref)
    ref=scipy.linalg.sqrtm(ref)
    mat=np.matmul(ref,np.matmul(dm,ref))
    return np.real(np.trace(scipy.linalg.sqrtm(mat)))**2
def chis_inFidelity(fw_states_T, objectives, tau_vals):
    ref=objectives[0].target.full()     
    return [qutip.Qobj(np.matmul(localTools.densityMatrix(ref),fw_states_T[0].full()))]
def J_T_inFidelity(fw_states_T,objectives,tau_vals=None,**kwargs):
    state=fw_states_T[0].full()
    ref=objectives[0].target.full()
    return inFidelity(state,localTools.densityMatrix(ref))
def inFidelity(state,ref=False):
    return 1-localFidelity(state,ref)
def functional_master(functional_name):
    if functional_name == 'inFidelity' or 'JT_ss':
        return [chis_inFidelity,J_T_inFidelity,inFidelity]

'''

