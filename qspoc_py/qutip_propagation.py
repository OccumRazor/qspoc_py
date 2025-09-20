import qutip,numpy as np,matplotlib.pyplot as plt
from . import task_obj,J_T_local,localTools

def plot_pulses(qutip_Ham,tlist):
    for i in range(len(qutip_Ham)):
        if isinstance(qutip_Ham[i],list):
            pulse = qutip_Ham[i][1](tlist,args=None)
            plt.plot(tlist,pulse)
    plt.show()

def qutip_pulse(fit_func):
    return lambda t,args:fit_func(t)

def qutip_Hamiltonian(Hamiltonian,pulse_options):
    Ham = []
    for i in range(len(Hamiltonian)):
        if isinstance(Hamiltonian[i],list):
            Ham.append([qutip.Qobj(Hamiltonian[i][0]),
                        qutip_pulse(pulse_options[Hamiltonian[i][1]]['args']['fit_func'])])
        else:
            Ham.append(qutip.Qobj(Hamiltonian[i]))
    return Ham

def qutip_c_ops(c_ops):
    new_ops = []
    for c_op in c_ops:
        new_ops.append(qutip.Qobj(c_op))
    return new_ops

import time

def qutip_prop_sg(prop:task_obj.Propagation):
    T = prop.tlist[0]
    nt = prop.tlist[1]*1
    tlist = localTools.half_step_tlist([T,nt])
    qutip_Ham = qutip_Hamiltonian(prop.Hamiltonian,prop.pulse_options)
    psi0 = prop.initial_states
    for i in range(len(psi0)):
        psi0[i] = qutip.Qobj(psi0[i])
    options = qutip.Options(store_final_state=True)
    qutip_psi_T = []
    t0 = time.time()
    for i in range(len(psi0)):
        result = qutip.mesolve(qutip_Ham,psi0[i],tlist,e_ops=[],c_ops=[],options=options)
        qutip_psi_T.append(result.final_state.full())
    t1 = time.time()
    return qutip_psi_T,t1-t0

def qutip_prop(prop:task_obj.Propagation,target_states,kappas=[],c_ops = []):
    gammas = [np.sqrt(kappa) for kappa in kappas]
    c_ops = qutip_c_ops(c_ops)
    T = prop.tlist[0]
    nt = prop.tlist[1]*1
    tlist = localTools.half_step_tlist([T,nt])
    psi_T = prop.propagate()
    JT = J_T_local.fidelity_mixed(target_states,psi_T)
    fidelity = [JT]
    purity = [1]
    qutip_Ham = qutip_Hamiltonian(prop.Hamiltonian,prop.pulse_options)
    #plot_pulses(qutip_Ham,tlist)
    psi0 = prop.initial_states
    for i in range(len(psi0)):
        psi0[i] = qutip.Qobj(psi0[i])
    options = qutip.Options(store_final_state=True)
    for gamma in gammas:
        qutip_psi_T = []
        if isinstance(gamma,np.ndarray):
            gamma_1 = list(gamma)
            if len(target_states) == 4:gamma_1 *= 2
            c_ops_gamma = [c_op * gamma_1i for c_op,gamma_1i in zip(c_ops,gamma_1)]
        else:c_ops_gamma = [c_op * gamma for c_op in c_ops]
        for i in range(len(psi0)):
            result = qutip.mesolve(qutip_Ham,psi0[i],tlist,e_ops=[],c_ops=c_ops_gamma,options=options)
            qutip_psi_T.append(result.final_state.full())
        JT = J_T_local.fidelity_mixed(target_states,qutip_psi_T)
        fidelity.append(JT)
        purity.append(sum([np.real(np.trace(np.matmul(qutip_psi_T[i],qutip_psi_T[i]))) for i in range(len(qutip_psi_T))])/len(qutip_psi_T))
    return fidelity,purity
