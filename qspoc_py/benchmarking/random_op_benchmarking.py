from config_jobs import config_prop_jobs
from qspoc_py import qutip_propagation,task_obj
import time

def table_making(data_dicts:dict):
    text = '\\hline\nn\\_setps '
    for key in data_dicts.keys():
        text += f'& {key} '
    text += '\\\\\n\\hline\n'
    for method in ['cheby','expm','qutip']:
        text += method
        for key in data_dicts.keys():
            text += f' & {data_dicts[key][method]:.3f}'
        text += f'\\\\\n\\hline\n'
    print(text)

if __name__ == '__main__':
    n_test = 10
    for gate in ['CX']:
        results = {}
        for T in [.1,.2,.3,.4,.5]:
            t_cheby = 0
            t_expm = 0
            t_qutip = 0
            n_steps = max(int(T*500),50)
            for _ in range(n_test):
                if gate in ['PE','CZ']:
                    n_cavities = 2
                else:
                    n_cavities = 1
                prop,target_state = config_prop_jobs(gate,[T,n_steps],prop_method='expm')
                t0 = time.time()
                psi_T_0 = prop.propagate()
                t1 = time.time()
                t_expm += t1 - t0
                prop.prop_method = 'cheby'
                prop.Hamiltonian = task_obj.sparse_Ham(prop.Hamiltonian)
                t0 = time.time()
                psi_T_1 = prop.propagate()
                t1 = time.time()
                t_cheby += t1 -t0
                qutip_psi_T,dt = qutip_propagation.qutip_prop_sg(prop)
                t_qutip += dt
            results[n_steps] = {'cheby':t_cheby/n_test,'expm':t_expm/n_test,'qutip':t_qutip/n_test}
        table_making(results)