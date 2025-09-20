import numpy as np,time,copy,matplotlib.pyplot as plt,random,time 
from . import localTools,read_write,propagation_method,fft_main,J_T_local,iter_info_manager,Weyl
from functools import partial
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

str2float_keys = ['oct_lambda_a','t_start','t_stop','t_rise','t_fall','oct_pulse_min','oct_pulse_max']
function_keys = ['args','update_shape']

def sort_property(ipt_list):
    text_content = ''
    n_dicts = len(ipt_list)
    if n_dicts == 1:
        return '\n* ' + ', '.join(f'{k} = {v}' for k,v in ipt_list[0].items()) + '\n\n'
    id_keys = [list(ipt_list[i].keys()) for i in range(n_dicts)]
    all_keys = list(ipt_list[0].keys())
    for i in range(1,n_dicts):all_keys += list(ipt_list[1].keys())
    all_keys = list(np.unique(all_keys))
    same_property = {}
    different_property = [{'ipt_list_id':i} for i in range(n_dicts)]
    for key in all_keys:
        if all([key in id_keys[i] for i in range(n_dicts)]):
            if all([ipt_list[0][key] == ipt_list[i][key] for i in range(n_dicts)]):
                same_property[key] = ipt_list[0][key]
            else:
                for i in range(n_dicts):
                    if key in id_keys[i]:
                        different_property[i][key] = ipt_list[i][key]
        else:
            for i in range(n_dicts):
                if key in id_keys[i]:
                    different_property[i][key] = ipt_list[i][key]
    text_content += ', '.join(f'{k} = {v}' for k,v in same_property.items()) + '\n'
    for i in range(n_dicts):
        different_property[i].pop('ipt_list_id')
        text_content += '* ' + ', '.join(f'{k} = {v}' for k,v in different_property[i].items()) + '\n'
    return text_content + '\n'

def dict2string(ipt_dict):
    text_content = ''
    for key in ipt_dict.keys():
        text_content += f'{key}: '
        if isinstance(ipt_dict[key],dict): text_content += ', '.join(f'{k} = {v}' for k,v in ipt_dict[key].items()) + '\n\n'
        elif isinstance(ipt_dict[key],list):
            text_content += sort_property(ipt_dict[key])
        else:
            raise KeyError('Anyway the format of the input dictionary does not match.')
    return text_content

def E_min_max(Hamiltonian,tlist,pulse_options):
    Hamiltonian = copy.deepcopy(Hamiltonian)
    assert isinstance(Hamiltonian,list) and len(Hamiltonian) > 0
    if isinstance(Hamiltonian[0],list):
        p_0 = Hamiltonian[0][1](tlist,pulse_options[Hamiltonian[0][1]]['args'])
        H_min = min(p_0) * Hamiltonian[0][0]
        H_max = max(p_0) * Hamiltonian[0][0]
    else:
        H_min = Hamiltonian[0]
        H_max = Hamiltonian[0]
    for H_i in Hamiltonian[1:]:
        if isinstance(H_i,list):
            p_i = H_i[1](tlist,pulse_options[H_i[1]]['args'])
            H_min += min(p_i) * H_i[0]
            H_min += max(p_i) * H_i[0]
        else:
            H_min += H_i
            H_max += H_i
    eig_vals_0 = eigsh(H_min,return_eigenvectors=False)
    eig_vals_1 = eigsh(H_max,return_eigenvectors=False)
    E_min = min([min(eig_vals_0),min(eig_vals_1)])
    E_max = max([max(eig_vals_0),max(eig_vals_1)])
    return E_max,E_min

def H_t(Hamiltonian,t,pulse_options,update_table=None):
    Hamiltonian = copy.deepcopy(Hamiltonian)
    assert isinstance(Hamiltonian,list) and len(Hamiltonian) > 0
    if isinstance(Hamiltonian[0],list):
        Ht = Hamiltonian[0][1](t,pulse_options[Hamiltonian[0][1]]['args']) * Hamiltonian[0][0]
    else:
        Ht = Hamiltonian[0]
    for H_i in Hamiltonian[1:]:
        if isinstance(H_i,list):
            if not update_table: 
                update_amp = 0
            else:
                update_amp = update_table[H_i[1]]
            Ht += (H_i[1](t,pulse_options[H_i[1]]['args']) + update_amp) * H_i[0]
        else:
            Ht += H_i
    return Ht

def sparsity(Hamiltonian):
    Hamiltonian = copy.deepcopy(Hamiltonian)
    assert isinstance(Hamiltonian,list) and len(Hamiltonian) > 0
    if isinstance(Hamiltonian[0],list):
        Hr = random.random() * Hamiltonian[0][0]
    else:
        Hr = Hamiltonian[0]
    for H_i in Hamiltonian[1:]:
        if isinstance(H_i,list):
            Hr += random.random() * H_i[0]
        else:
            Hr += H_i
    Hr = csr_matrix(Hr)
    return 1 - Hr.nnz/(Hr.shape[0] * Hr.shape[1])

def sparse_Ham(Hamiltonian):
    for i in range(len(Hamiltonian)):
        if isinstance(Hamiltonian[i],list):
            Hamiltonian[i][0] = csr_matrix(Hamiltonian[i][0])
        else:
            Hamiltonian[i] = csr_matrix(Hamiltonian[i])
    return Hamiltonian

class Propagation:
    def __init__(self,Hamiltonian,tlist,prop_method,initial_states,pulse_name,pulse_options = None):
        self.Hamiltonian = Hamiltonian
        self.tlist = tlist
        self.tlist_long = localTools.half_step_tlist(self.tlist)
        self.prop_method = prop_method
        self.n_states = len(initial_states)
        self.initial_states = initial_states
        self.pulse_name = pulse_name
        self.pulse_options = pulse_options
        self.c_ops = None
        self.shape_function()
        self.sparsity = sparsity(self.Hamiltonian)
        if self.sparsity > 0.85 and prop_method != 'expm':self.Hamiltonian = sparse_Ham(self.Hamiltonian)
        self.E_max,self.E_min = E_min_max(self.Hamiltonian,self.tlist_long,self.pulse_options)
    
    def add_dissipator(self,c_ops):
        self.c_ops = c_ops

    def shape_function(self):
        if len(self.tlist) == 2:
            ipt_tlist = [0,self.tlist[0],self.tlist[1]]
        if len(self.tlist) == 3:
            ipt_tlist = self.tlist
        for k,v in self.pulse_options.items():
            if 'oct_lambda_a' in v.keys():
                if 'oct_shape' not in v.keys():
                    self.pulse_options[k]['oct_shape'] = 'flattop'
                if self.pulse_options[k]['oct_shape'] == 'flattop':
                    self.pulse_options[k]['update_shape']=partial(localTools.S,t_start=ipt_tlist[0], t_stop=ipt_tlist[1], t_rise=self.pulse_options[k]['t_rise'],t_fall=self.pulse_options[k]['t_fall'])

    def change_lambda_a(self,change_factor,approach = 1):
        if approach:
            for k in self.pulse_options.keys():
                self.pulse_options[k]['oct_lambda_a'] *= change_factor
        else:
            for k in self.pulse_options.keys():
                self.pulse_options[k]['oct_lambda_a'] = change_factor

    def check_pulse(self,t,pulse_i,control_i_update_amp):
        if all(['oct_pulse_min' not in self.pulse_options[pulse_i].keys(),'oct_pulse_max' not in self.pulse_options[pulse_i].keys()]):
            return control_i_update_amp
        pulse_i_t = control_i_update_amp+pulse_i(t,self.pulse_options[pulse_i]['args'])
        if 'oct_pulse_max' in self.pulse_options[pulse_i].keys():
            #assert pulse_i(t,self.pulse_options[pulse_i]['args']) <= self.pulse_options[pulse_i]['oct_pulse_max'] + 1e-9, f'AssertionErrot at t = {t
            #        }, pulse_i(t) = {pulse_i(t,self.pulse_options[pulse_i]['args'])}, oct_pulse_max: {self.pulse_options[pulse_i]['oct_pulse_max']}'
            if pulse_i_t > self.pulse_options[pulse_i]['oct_pulse_max']:
                control_i_update_amp -= pulse_i_t - self.pulse_options[pulse_i]['oct_pulse_max']
        if 'oct_pulse_min' in self.pulse_options[pulse_i].keys():
            #assert pulse_i(t,self.pulse_options[pulse_i]['args']) >= self.pulse_options[pulse_i]['oct_pulse_min'] - 1e-9, f'AssertionErrot at t = {t
            #        }, pulse_i(t) = {pulse_i(t,self.pulse_options[pulse_i]['args'])}, oct_pulse_min: {self.pulse_options[pulse_i]['oct_pulse_min']}'
            if pulse_i_t < self.pulse_options[pulse_i]['oct_pulse_min']:
                control_i_update_amp -= pulse_i_t - self.pulse_options[pulse_i]['oct_pulse_min']
        return control_i_update_amp

    def propagate_sg(self,dt,t,psi_0,backwards=False):
        psi_0 = copy.deepcopy(psi_0)
        Ht = H_t(self.Hamiltonian,t,self.pulse_options)
        for i in range(self.n_states):
            if self.prop_method == 'cheby':
                psi_0[i] = propagation_method.Chebyshev(Ht,psi_0[i],self.E_max,self.E_min,dt,backwards=backwards)
            if self.prop_method == 'expm':
                psi_0[i] = propagation_method.Matrix_Exponential(Ht,psi_0[i],dt)
        return psi_0

    def propagate_sg_update(self,dt,t,psi_0,chis):
        psi_0 = copy.deepcopy(psi_0)
        update_table = {}
        state_size = psi_0[0].size
        for i in range(self.n_states):
            chis[i] = np.reshape(chis[i],(state_size))
        update_return = {}
        ga_return = []
        for k in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[k],list):
                control_k_update_amp = 0
                Hk = self.Hamiltonian[k][0]
                pulse_k = self.Hamiltonian[k][1]
                update_shape_k_t = self.pulse_options[pulse_k]['update_shape'](t)
                oct_lambda_a_k = self.pulse_options[pulse_k]['oct_lambda_a']
                if oct_lambda_a_k and update_shape_k_t != 0:
                    for i in range(self.n_states):
                        Hk_psi = Hk.dot(psi_0[i])
                        control_k_update_amp += np.linalg.norm(chis[i],2) * np.imag(np.inner(np.conjugate(chis[i]),np.reshape(Hk_psi,(state_size))))
                    control_k_update_amp *= update_shape_k_t / oct_lambda_a_k
                    control_k_update_amp = self.check_pulse(t,pulse_k,control_k_update_amp)
                    update_table[pulse_k] = control_k_update_amp
                    ga_return.append(control_k_update_amp * dt)
                    update_return[pulse_k] = control_k_update_amp+pulse_k(t,self.pulse_options[pulse_k]['args'])
                else:
                    update_table[pulse_k] = 0
                    ga_return.append(0)
                    update_return[pulse_k] = pulse_k(t,self.pulse_options[pulse_k]['args'])
        Ht = H_t(self.Hamiltonian,t,self.pulse_options,update_table)
        for i in range(self.n_states):
            psi_0[i] = propagation_method.Chebyshev(Ht,psi_0[i],self.E_max,self.E_min,dt)
        return psi_0,update_return,ga_return

    def propagate(self,backwards=False,store_states = False,update=False,chis_t=None,prop_options: dict = {}):
        if 'initial_states' in prop_options.keys():psi_0 = copy.deepcopy(prop_options['initial_states'])
        else:psi_0 = copy.deepcopy(self.initial_states)
        dt = self.tlist_long[1] - self.tlist_long[0]
        prop_tlist = copy.deepcopy(self.tlist_long)
        if backwards:prop_tlist = np.flip(prop_tlist,0)
        if store_states:psi_t = [psi_0]
        if update:
            new_controls = {}
            for H_i in self.Hamiltonian:
                if isinstance(H_i,list):
                    new_controls[H_i[1]] = np.zeros(len(prop_tlist))
            ga_int = np.zeros(len(self.pulse_options))
        for i in range(len(prop_tlist)):
            t = prop_tlist[i]
            if update:
                psi_0,update_return,ga_return = self.propagate_sg_update(dt,t,psi_0,chis_t[i])
                for k in range(len(new_controls)):
                    ga_int[k] += np.abs(ga_return[k])
                for H_i in self.Hamiltonian:
                    if isinstance(H_i,list):
                        new_controls[H_i[1]][i] = update_return[H_i[1]]
            else:psi_0 = self.propagate_sg(dt,t,psi_0,backwards=backwards)
            if store_states:psi_t.append(psi_0)
        if update:
            ga_int = sum(ga_int)/len(ga_int)
            return psi_0,new_controls,ga_int
        else:
            if store_states:return psi_t
            else:return psi_0

    def update_control(self,new_controls):
        for H_i in self.Hamiltonian:
            if isinstance(H_i,list):
                if H_i[1] in new_controls.keys():
                    self.pulse_options[H_i[1]]['args']["fit_func"] =  interp1d(
                    self.tlist_long, new_controls[H_i[1]], kind="cubic", fill_value="extrapolate")
        self.E_max,self.E_min = E_min_max(self.Hamiltonian,self.tlist_long,self.pulse_options)

    def obtain_pulse(self):
        pulses = {}
        for Hi in self.Hamiltonian:
            if isinstance(Hi,list):
                pulses[Hi[1]] = self.pulse_options[Hi[1]]['args']["fit_func"]
        return pulses

    def plot_pulses(self,ipt_controls = None):
        if ipt_controls:
            for i in range(len(ipt_controls)):
                plt.plot(self.tlist_long,ipt_controls[i],label = f'ipt_pulse_{i}')
            plt.legend(loc='best')
            plt.show()
        else:
            for i in range(len(self.Hamiltonian)):
                if isinstance(self.Hamiltonian[i],list):
                    plt.plot(self.tlist_long,self.Hamiltonian[i][1](self.tlist_long,self.pulse_options[self.Hamiltonian[i][1]]['args']),label=f'pulse {i}')
            plt.legend(loc='best')
            plt.show()

    def config_prop(self,path,write = False,zero_base = True):
        path_Path = Path(path)
        path_Path.mkdir(exist_ok=True,parents=True)
        if len(self.tlist) == 2:cfg_tlist = [0,self.tlist[0],self.tlist[1]]
        else:cfg_tlist = [self.tlist[0],self.tlist[1],self.tlist[2]]
        config_dict = {'prop':{'prop_method':self.prop_method},'tgrid':{'t_start':cfg_tlist[0],'t_stop':cfg_tlist[1],'nt':cfg_tlist[2]}}
        Hamiltonian_info = []
        pulse_info = []
        pulse_count = 0
        for i in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[i],list):
                id_pulse_option = self.pulse_options[self.Hamiltonian[i][1]]
                mat_text = read_write.matrix2text(self.Hamiltonian[i][0],zero_base= zero_base)
                pulse_text = read_write.control2text(self.tlist_long,self.Hamiltonian[i][1](self.tlist_long,id_pulse_option['args']))
                with open(path + f'{self.pulse_name}_{pulse_count}.dat','w') as pulse_file:
                    pulse_file.write(pulse_text)
                Hamiltonian_info.append({'dim':self.Hamiltonian[i][0].shape[0],'filename':f'H{i}.dat','pulse_id':pulse_count})
                id_pulse_info = {'pulse_id':pulse_count,'filename':f'{self.pulse_name}_{pulse_count}.dat'}
                for key in id_pulse_option.keys():
                    if key not in function_keys:
                        id_pulse_info[key] = id_pulse_option[key]
                pulse_info.append(id_pulse_info)
                pulse_count += 1
            else:
                mat_text = read_write.matrix2text(self.Hamiltonian[i],zero_base = zero_base)
                Hamiltonian_info.append({'dim':self.Hamiltonian[i].shape[0],'filename':f'H{i}.dat'})
            with open(path + f'H{i}.dat', 'w') as mat_file:
                mat_file.write(mat_text)
        psi_info = []
        if self.n_states == 1:
            state_text = read_write.state2text(self.initial_states[0],zero_base = zero_base)
            with open(path + 'psi_initial.dat','w') as state_file:
                state_file.write(state_text)
            psi_info.append({'filename':'psi_initial.dat','label':'initial'})
        else:
            for i in range(self.n_states):
                state_text = read_write.state2text(self.initial_states[i],zero_base = zero_base)
                with open(path + f'psi_initial_{i}.dat','w') as state_file:
                    state_file.write(state_text)
                psi_info.append({'filename':f'psi_initial_{i}.dat','label':'initial'})
        config_dict['ham'] = Hamiltonian_info
        config_dict['pulse'] = pulse_info
        config_dict['psi'] = psi_info
        if write:
            config_text = dict2string(config_dict)
            with open(path + 'config', 'w') as config_file:
                config_file.write(config_text)
        return config_dict

class Optimization(Propagation):
    def __init__(self,Hamiltonian,tlist,prop_method,initial_states,pulse_name,pulse_options):
        super().__init__(Hamiltonian,tlist,prop_method,initial_states,pulse_name,pulse_options)
    
    def custom_init(self,opt_method,JT_conv,delta_JT_conv,iter_dat,iter_stop):
        self.oct_info = {
            'oct_method':opt_method,
            'JT_conv':JT_conv,
            'delta_JT_conv':delta_JT_conv,
            'iter_dat':iter_dat,
            'iter_stop':iter_stop}
        self.target_states = None
        self.observables = None
        self.functional_info = None
        self.psi_T_analysis = None
        self.n_JT = 1
        self.JT_name = None

    def set_target_states(self,target_states,lambda_U=0):
        self.target_states = target_states
        if lambda_U:
            self.JT = J_T_local.JT_Phi3(self.target_states,self.initial_states,lambda_U)
            self.chis = J_T_local.chis_Phi3(self.target_states,self.initial_states,lambda_U)
            self.functional_info = f'Functional name: PE\nFunctional parameters: lambda_U = {lambda_U}'
            self.JT_name = ['JT','tau_re','delta_U']
            self.n_JT = 3
        else:
            #self.JT = partial(J_T_local.JT_ss,self.target_states)
            #self.chis = partial(J_T_local.chis_ss,self.target_states)
            self.JT = partial(J_T_local.JT_re,self.target_states)
            self.chis = partial(J_T_local.chis_re,self.target_states)
            self.n_JT = 1
    
    def set_gate_objectives(self,basis_states,gate):
        self.JT = J_T_local.JT_gate(basis_states,gate)
        self.chis = 0

    def set_observables(self,observables):
        self.observables = observables

    def set_PE_objectives(self,basis,lambda_U = 0.5):
        Bell_basis_states = [np.sqrt(0.5) * (self.initial_states[0] + self.initial_states[3]),
                             1j * np.sqrt(0.5) * (self.initial_states[0] - self.initial_states[3]),
                             1j * np.sqrt(0.5) * (self.initial_states[1] + self.initial_states[2]),
                             np.sqrt(0.5) * (self.initial_states[1] - self.initial_states[2])]
        self.initial_states = Bell_basis_states
        self.JT,self.psi_T_analysis = J_T_local.JT_PE(Bell_basis_states,lambda_U)
        self.chis = J_T_local.chis_PE(basis,lambda_U)
        self.functional_info = f'Functional name: PE\nFunctional parameters: lambda_U = {lambda_U}'
        self.JT_name = ['JT','conc','delta_U']
        self.n_JT = 3

    def config_opt(self,path,zero_base = True):
        path_Path = Path(path)
        path_Path.mkdir(exist_ok=True,parents=True)
        config_dict = self.config_prop(path,False,zero_base)
        if self.target_states:
            psi_info = []
            if self.n_states > 1:
                for i in range(len(self.target_states)):
                    psi_text = read_write.state2text(self.target_states[i],zero_base = zero_base)
                    with open(path+f'psi_final_{i}.dat', 'w') as psi_file:
                        psi_file.write(psi_text)
                    psi_info.append({'filename':f'psi_final_{i}.dat','label':'final'})
            else: 
                psi_text = read_write.state2text(self.target_states[0],zero_base = zero_base)
                with open(path+f'psi_final.dat', 'w') as psi_file:
                    psi_file.write(psi_text)
                psi_info.append({'filename':f'psi_final.dat','label':'final'})
            config_dict['psi'] += psi_info
        if self.observables:
            observables_dict = []
            for i in range(len(self.observables)):
                observable_text = read_write.matrix2text(self.observables[i],zero_base = zero_base)
                with open(path+f'O{i}.dat', 'w') as observable_file:
                    observable_file.write(observable_text)
                observables_dict.append({'filename':f'O{i}.dat'})
            config_dict['observables'] = observables_dict
        config_dict['oct'] = self.oct_info
        config_text = dict2string(config_dict)
        with open(path + 'config', 'w') as config_file:
            config_file.write(config_text)

    def Krotov_optimization(self,runfolder = None, monotonic = False):
        if runfolder:
            self.config_opt(runfolder)
        opt_result_options = iter_info_manager.Opt_result_options(False,False,'last')
        opt_result = iter_info_manager.Opt_result(self.oct_info['iter_stop'],self.tlist_long,self.Hamiltonian,self.pulse_options,opt_result_options,runfolder,self.n_JT,self.JT_name,False,self.functional_info)
        opt_result.store_initial_controls()
        JT_iter = []
        tic = time.time()
        psi_T = self.propagate()
        psi_T_last_step = copy.deepcopy(psi_T)
        tac = time.time()
        JT_0 = self.JT(psi_T)
        if not isinstance(JT_0,list):JT_0 = [JT_0]
        opt_result.log_iter_info(0,JT_0,tac-tic,0,0)
        JT_iter.append(JT_0[0])
        ga_bound = 10000
        for iters in range(1,self.oct_info['iter_stop'] + 1):
            chis_T = self.chis(psi_T)
            tic = time.time()
            chis_t = self.propagate(True,True,False,prop_options={'initial_states':chis_T})
            chis_t.reverse()
            psi_T,new_controls,ga_int = self.propagate(False,False,True,chis_t)
            tac = time.time()
            for H_i in self.Hamiltonian:
                if isinstance(H_i,list):
                    if self.pulse_options[H_i[1]]['oct_lambda_a'] and 'fft_threshold' in self.pulse_options[H_i[1]].keys():
                        new_controls[H_i[1]] = fft_main.fft_filter(self.tlist_long,new_controls[H_i[1]],self.pulse_options[H_i[1]]['fft_threshold'])
            psi_T_last_step = copy.deepcopy(psi_T)
            JT_new = self.JT(psi_T)
            if not isinstance(JT_new,list):JT_new = [JT_new]
            if all([JT_new[0] > JT_iter[-1] and monotonic]) or ga_int > ga_bound:
                opt_result.log_break_info(JT_new,JT_iter[-1],iters,ga_int,ga_bound)
                psi_T = copy.deepcopy(psi_T_last_step)
                self.change_lambda_a(2)
            else:
                opt_result.store_psi_T(psi_T)
                opt_result.log_iter_info(iters,JT_new,tac-tic,JT_iter[-1],ga_int)
                JT_iter.append(JT_new[0])
                self.update_control(new_controls)
                opt_result.store_control(new_controls)
                if JT_iter[-1] < self.oct_info['JT_conv'] or np.abs(JT_iter[-2] - JT_iter[-1]) < self.oct_info['delta_JT_conv']:
                    #opt_result.log_stop_info(JT_iter[-1],self.oct_info['JT_conv'],np.abs(JT_iter[-2] - JT_iter[-1]),self.oct_info['delta_JT_conv'])
                    opt_result.log_stop_info(JT_new,self.oct_info['JT_conv'],np.abs(JT_iter[-2] - JT_iter[-1]),self.oct_info['delta_JT_conv'])
                    break
        if self.psi_T_analysis:
            opt_result.log_psi_T_analysis(self.psi_T_analysis)
        return opt_result

    def GRAPE_update_pulse(self,psi_t,lambda_t,Hamiltonian,tlist,n_states,pulse_options):
        lambda_t.reverse()
        state_size = psi_t[0][0].size
        for i in range(len(tlist)):
            for j in range(self.n_states):
                lambda_t[i][j] = np.reshape(lambda_t[i][j],(state_size))
        #for i in range(len(lambda_t)):
            #for k in range(n_states):
                #psi_t[i][k] = localTools.densityMatrix(psi_t[i][k])
                #lambda_t[i][k] = localTools.densityMatrix(lambda_t[i][k])
        new_pulses = {}
        dt = tlist[1] - tlist[0]
        ga = []
        #for i in range(len(Hamiltonian)):
        for H_i in Hamiltonian:
            if isinstance(H_i,list):
                oct_lambda_a = pulse_options[H_i[1]]['oct_lambda_a']
                if oct_lambda_a:
                    epsilon = 1 / oct_lambda_a
                    id_pulse = []
                    id_ga = 0
                    for j in range(len(tlist)):
                        Delta_it = 0
                        for k in range(n_states):
                            Hi_psi = H_i[0].dot(psi_t[j][k])
                            Delta_it += 2 * dt * np.imag(np.inner(np.conjugate(lambda_t[j][k]),np.reshape(Hi_psi,(state_size))))
                            #Delta_it += np.real(-1j * dt * np.trace(np.matmul(lambda_t[j][k],
                            #            np.matmul(H_i[0],psi_t[j][k])-np.matmul(psi_t[j][k],H_i[0]))))
                        update_amp = pulse_options[H_i[1]]['update_shape'](tlist[j])*epsilon*Delta_it
                        update_amp = self.check_pulse(tlist[j],H_i[1],update_amp)
                        id_pulse.append(H_i[1](tlist[j],pulse_options[H_i[1]]['args'])+update_amp)
                        id_ga += np.abs(update_amp)
                    new_pulses[H_i[1]] = id_pulse
                    ga.append(id_ga)
                else:
                    new_pulses[H_i[1]] = H_i[1](tlist,pulse_options[H_i[1]]['args'])
                    ga.append(0)
        return new_pulses,sum(ga)/len(ga)

    def GRAPE(self,runfolder = None, monotonic = False):
        if runfolder:
            self.config_opt(runfolder)
        opt_result_options = iter_info_manager.Opt_result_options(False,False,'last')
        oct_direction = False # Gradient Descent if not True else Gradient Ascent
        opt_result = iter_info_manager.Opt_result(self.oct_info['iter_stop'],self.tlist_long,self.Hamiltonian,self.pulse_options,opt_result_options,runfolder,self.n_JT,self.JT_name,oct_direction,self.functional_info)
        opt_result.store_initial_controls()
        JT_iter = []
        for iters in range(self.oct_info['iter_stop']):
            tic_0 = time.time()
            psi_t = self.propagate(False,True)
            psi_T = psi_t[-1]
            tac_0 = time.time()
            JT_new = self.JT(psi_T)
            if not isinstance(JT_new,list):JT_new = [JT_new]
            if iters == 0:
                opt_result.log_iter_info(0,JT_new,tac_0-tic_0,0,0)
            else:
                opt_result.log_iter_info(iters,JT_new,tac_0-tic_1,JT_iter[-1],ga_int)
            opt_result.store_psi_T(psi_T)
            if iters:
                monotonicity_break = JT_iter[-1] > JT_new[0] if oct_direction else JT_iter[-1] < JT_new[0]
                if monotonicity_break and monotonic:
                    opt_result.log_break_info(JT_new,JT_iter[-1],iters,ga_int,ga_bound = 1e10)
                    self.change_lambda_a(2)
            tic_1 = time.time()
            lambda_t = self.propagate(True,True,prop_options={'initial_states':self.chis(psi_T)})
            new_pulses,ga_int = self.GRAPE_update_pulse(psi_t,lambda_t,self.Hamiltonian,self.tlist_long,self.n_states,self.pulse_options)
            self.update_control(new_pulses)
            opt_result.store_control(new_pulses)
            JT_iter.append(JT_new[0])
            if iters:
                if JT_iter[-1] < self.oct_info['JT_conv'] or np.abs(JT_iter[-1] - JT_iter[-2]) < self.oct_info['delta_JT_conv']:
                    opt_result.log_stop_info(JT_new,self.oct_info['JT_conv'],np.abs(JT_iter[-2] - JT_iter[-1]),self.oct_info['delta_JT_conv'])
                    break
        iters += 1
        psi_T = self.propagate(False,False)
        tac_2 = time.time()
        opt_result.store_psi_T(psi_T)
        JT_new = self.JT(psi_T)
        if not isinstance(JT_new,list):JT_new = [JT_new]
        opt_result.log_iter_info(iters,JT_new,tac_2-tic_1,JT_iter[-1],ga_int)
        if self.psi_T_analysis:
            opt_result.log_psi_T_analysis(self.psi_T_analysis)
        JT_iter.append(JT_new[0])
        return opt_result

    def GRAPE_Grad(self,psi_t,lambda_t,order=2):
        '''
        Order: int should equal 
        '''
        assert order in [1,2], f'order should euqal either 1 or 2. Input order: {order}'
        psi_t = copy.deepcopy(psi_t)
        lambda_t = copy.deepcopy(lambda_t)
        lambda_t.reverse()
        state_size = psi_t[0][0].size
        dt = self.tlist_long[1] - self.tlist_long[0]
        h = 1e-10
        grad = []
        update_table = {}
        for i in range(len(self.Hamiltonian)):
            if isinstance(self.Hamiltonian[i],list):
                update_table[self.Hamiltonian[i][1]] = 0
        for H_i in self.Hamiltonian:
            if isinstance(H_i,list):
                pulse_i = H_i[1]
                if self.pulse_options[pulse_i]['oct_lambda_a']:
                    for i in range(self.tlist[-1] - 1):
                        update_table[pulse_i] = h
                        Hip = H_t(self.Hamiltonian,self.tlist_long[i],self.pulse_options,update_table)
                        if order == 2:
                            update_table[pulse_i] = -h
                            Him = H_t(self.Hamiltonian,self.tlist_long[i],self.pulse_options,update_table)
                        update_table[pulse_i] = 0
                        grad_i = 0.
                        for j in range(self.n_states):
                            if order == 2:
                                Hi_psi = propagation_method.Chebyshev(Hip,psi_t[i][j],self.E_max,self.E_min,dt) - propagation_method.Chebyshev(Him,psi_t[i][j],self.E_max,self.E_min,dt)
                            else:
                                Hi_psi = propagation_method.Chebyshev(Hip,psi_t[i][j],self.E_max,self.E_min,dt) - psi_t[i+1][j]
                            grad_i -= 1/h/order *  np.real(np.inner(np.conjugate(np.reshape(lambda_t[i][j],(state_size))),np.reshape(Hi_psi,(state_size))))
                        grad.append(grad_i)
        return grad

    def GRAPE_BFGS(self,runfolder):
        from scipy.optimize import fmin_l_bfgs_b
        def func(x,approx_grad,order,*args):
            new_controls = localTools.array2control(x,self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
            self.update_control(new_controls)
            psi_t = self.propagate(store_states=True)
            if not approx_grad:
                lambda_T = self.chis(psi_t[-1])
                lambda_t = self.propagate(True,True,prop_options={'initial_states':lambda_T})
                grad = self.GRAPE_Grad(psi_t,lambda_t,order)
            JT_eval = self.JT(psi_t[-1])
            if not isinstance(JT_eval,list):JT_eval = [JT_eval]
            if approx_grad:return JT_eval,psi_t[-1]
            else:return JT_eval,np.real(grad),psi_t[-1]
        if runfolder:
            self.config_opt(runfolder)
        x0 = localTools.control2array(self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
        log_options = iter_info_manager.Opt_result_options(False,False,'last')
        scipy_monitor = iter_info_manager.Monitor(self.oct_info['iter_stop'],self.tlist,self.tlist_long,self.Hamiltonian,self.pulse_options,log_options,runfolder,self.n_JT,self.JT_name,self.functional_info,func,x0, order = 2)
        scipy_monitor.store_initial_controls()
        bounds = localTools.array_bounds(self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
        scipy_monitor.set_iter_params(factr = 1e4, maxls = 40,pgtol = 1e-8)
        x,f,d = fmin_l_bfgs_b(scipy_monitor.cost_function,x0,approx_grad=scipy_monitor.approx_grad,bounds = bounds,maxiter=scipy_monitor.iter_stop,
                              maxfun=scipy_monitor.maxfun,callback=scipy_monitor.callback,factr=scipy_monitor.factr,maxls=scipy_monitor.maxls,pgtol=scipy_monitor.pgtol)
        new_controls = localTools.array2control(x,self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
        self.update_control(new_controls)
        scipy_monitor.log_finish_info(x,f,d)
        if self.psi_T_analysis:
            scipy_monitor.log_psi_T_analysis(self.psi_T_analysis)
        return scipy_monitor
    
    def Nelder_Mead(self,runfolder,options):
        if 'n_params' in options.keys():
            n_params = options['n_params']
        else:
            n_params = 15
        from scipy.optimize import minimize
        method = 'Nelder-Mead'
        def func(x):
            new_controls = localTools.Crab_pulse(x,n_params,n_pulses,c0j,self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
            self.update_control(new_controls)
            psi_T = self.propagate()
            JT_eval = J_T_local.reverse_GME_concurrence(psi_T)
            if not isinstance(JT_eval,list):JT_eval = [JT_eval]
            return JT_eval,psi_T
        if runfolder:
            self.config_opt(runfolder)
        n_pulses = 0
        for H_i in self.Hamiltonian:
            if isinstance(H_i,list):
                pulse_i = H_i[1]
                if self.pulse_options[pulse_i]['oct_lambda_a']:
                    n_pulses += 1
        c0j = copy.deepcopy(self.pulse_options)
        x0 = localTools.gen_Crab_parameters(n_params,n_pulses)
        log_options = iter_info_manager.Opt_result_options(False,False,'last')
        scipy_monitor = iter_info_manager.Monitor_Nelder_Mean(self.oct_info['iter_stop'],self.tlist,self.tlist_long,self.Hamiltonian,self.pulse_options,log_options,runfolder,self.n_JT,self.JT_name,self.functional_info,func,x0,c0j,n_params,n_pulses)
        scipy_monitor.store_initial_controls()
        options = {'maxiter':self.oct_info['iter_stop'] * (n_params + 1),'maxfev':self.oct_info['iter_stop'] * (n_params + 1)}
        bounds = localTools.Crab_bounds(n_params,n_pulses)
        result = minimize(scipy_monitor.cost_function,x0,method=method,bounds = bounds,
                              callback=scipy_monitor.callback,options=options)
        new_controls = localTools.Crab_pulse(result.x,n_params,n_pulses,c0j,self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
        self.update_control(new_controls)
        scipy_monitor.log_finish_info(result)
        if self.psi_T_analysis:
            scipy_monitor.log_psi_T_analysis(self.psi_T_analysis)
        return scipy_monitor
        

    def optimize(self,runfolder = None, monotonic = False,options = None):
        if self.oct_info['oct_method'] == 'Krotov':opt_result = self.Krotov_optimization(runfolder,monotonic)
        if self.oct_info['oct_method'] == 'GRAPE':opt_result = self.GRAPE(runfolder,monotonic)
        if self.oct_info['oct_method'] == 'GRAPE-BFGS':opt_result = self.GRAPE_BFGS(runfolder)
        if self.oct_info['oct_method'] == 'Nelder-Mead':opt_result = self.Nelder_Mead(runfolder,options=options)
        return opt_result