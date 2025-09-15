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

def qutip_prop(prop:task_obj.Propagation,target_states,mus=[],c_ops = []):
    c_ops = qutip_c_ops(c_ops)
    T = prop.tlist[0]
    nt = prop.tlist[1]*1
    tlist = localTools.half_step_tlist([T,nt])
    psi_T = prop.propagate()
    JT = J_T_local.fidelity_mixed(target_states,psi_T)
    print(JT)
    qutip_Ham = qutip_Hamiltonian(prop.Hamiltonian,prop.pulse_options)
    #plot_pulses(qutip_Ham,tlist)
    psi0 = prop.initial_states
    options = qutip.Options(store_final_state=True)
    for mu in mus:
        qutip_psi_T = []
        for i in range(len(psi0)):
            psi0[i] = qutip.Qobj(psi0[i])
            if mu:result = qutip.mesolve(qutip_Ham,psi0[i],tlist,e_ops=[],c_ops=c_ops,options=options)
            else:result = qutip.mesolve(qutip_Ham,psi0[i],tlist,e_ops=[],c_ops=[],options=options)
            qutip_psi_T.append(result.final_state.full())
        JT = J_T_local.fidelity_mixed(target_states,qutip_psi_T)
        print(JT)
        #print(result.solver)
