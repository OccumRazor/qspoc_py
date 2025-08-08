import time,numpy as np,matplotlib.pyplot as plt,datetime
from pathlib import Path
from scipy.interpolate import interp1d
from dataclasses import dataclass
from . import read_write,localTools

class Iter_info:
    def __init__(self,iter_stop:int,runfolder=None,n_JT:int=1,JT_names=None,direction:bool=False,functional_info=None):
        '''
        Parameters
        ----------
        iter_stop: int specify number of iterations, required for the purpose of alignment.\n
        runfolder: upon input, iter info will be written into runfolder/oct_iters.dat, otherwise printed to terminal.\n
        n_JT: give n_JT > 1 if JT is composed of multiple terms and each term is given specifically, otherwise only the first digit will be logged.\n
        JT_names: list containing name of each JT term, if n_JT > 1 and JT_names == None, it willed be setted to ['JT'] + [f'JT_{i}' for i in range(n_JT - 1)]\n
        direction: == 1 means gradient ascent (-like)
        '''
        self.n_JT = n_JT
        self.iter_stop = iter_stop
        self.runfolder = runfolder
        self.iter_str_len = len(str(iter_stop)) + 2
        self.direction = direction
        self.JT_iter = []
        if not JT_names:
            self.JT_names = ['JT'] + [f'JT_{i}' for i in range(n_JT-1)]
        else:self.JT_names = JT_names
        if self.runfolder:
            self.out_stream = open(runfolder + 'oct_iters.dat','w')
        message = f'#{' ' * (self.iter_str_len - 3)}iter{' ' * 4}'
        #for JT_name in self.JT_names:
        #    message += JT_name + ' ' * (14 - len(JT_name))
        #message += f' dJT{' ' * 11}ga_int{' ' * 8}dt'
        for JT_name in self.JT_names:
            message += JT_name + ' ' * (18 - len(JT_name))
        message += f' dJT{' ' * 15}ga_int{' ' * 12}dt'
        if self.runfolder:
            if functional_info:
                with open(runfolder + 'functional_info.dat','w') as functional_log:
                    functional_log.write(functional_info)
            self.out_stream.write(message+'\n')
        else:
            print(functional_info)
            print(message)
    
    def log_iter_info(self,iters,JT_new,dt,JT_last=None,ga_int=None):
        dJT_sign_placeholder = ' '
        self.JT_iter.append(JT_new)
        if not JT_last:
            dJT = 0.0
        else:
            dJT = JT_last - JT_new[0]
            if self.direction:
                dJT = -dJT
        if dJT<0:dJT_sign_placeholder = ''
        if not ga_int:ga_int = 0.0
        message = f'{' ' * (self.iter_str_len-len(str(iters))+2)}{iters}{' ' * 4}'
        #for i in range(self.n_JT): message += f'{JT_new[i]:.8f}{' ' * 4}'
        #message += f'{dJT:.8f}{' ' * 4}{ga_int:.8f}{' ' * 4}{dt:.2f}'
        for i in range(self.n_JT): message += f'{JT_new[i]:.8e}{' ' * 4}'
        message += f'{dJT_sign_placeholder}{dJT:.8e}{' ' * 4}{ga_int:.8e}{' ' * 4}{dt:.2f}'
        if self.runfolder:
            self.out_stream.write(message+'\n')
            self.out_stream.flush()
        else:print(message)
    
    def log_break_info(self,JT_new,JT_last,iters,ga_int,ga_bound):
        if self.direction:
            if JT_new[0] < JT_last:message = f'#{' ' * (1 + self.iter_str_len - len(str(iters)))}{iters} monotonicity breaks, JT_new = {JT_new[0]}, increase lambda_a by a factor of 2.'
            else:message = f'# ga_int ({ga_int}) > ga_bound ({ga_bound}), increase lambda_a by a factor of 2.'
        else:
            if JT_new[0] > JT_last:message = f'#{' ' * (1 +self.iter_str_len - len(str(iters)))}{iters} monotonicity breaks, JT_new = {JT_new[0]}, increase lambda_a by a factor of 2.'
            else:message = f'# ga_int ({ga_int}) > ga_bound ({ga_bound}), increase lambda_a by a factor of 2.'
        if self.runfolder:
            self.out_stream.write(message+'\n')
            self.out_stream.flush()
        else:print(message)

    def log_stop_info(self,JT_new,JT_conv,dJT,dJT_conv):
        if self.direction:
            message = f'# stop condition met (JT > {JT_conv}: {JT_new[0] > JT_conv}, dJT < {dJT_conv}: {dJT < dJT_conv}), break'
        else:
            message = f'# stop condition met (JT < {JT_conv}: {JT_new[0] < JT_conv}, dJT < {dJT_conv}: {dJT < dJT_conv}), break'
        if self.runfolder:self.out_stream.write(message+'\n')
        else:print(message)

    def log_time2out_file(self,keyword:str=''):
        message = f'# entrance time: {datetime.datetime.now()}\n# {keyword}'
        if self.runfolder:
            self.out_stream.write(message+'\n')
            self.out_stream.flush()
        else:print(message)

@dataclass
class Opt_result_options:
    '''
    store_psi_T_iter:\n
        if True, store psi_T of each iteration, instead of the last one.\n
    sotre_intermideate_state:\n
        if True, store pst_t instead of just pst_T\n
    store_former_control_key:\n
        last: only store the controls from the last two iterations. Initial control is not controled by this key.\n
        all: store all controls.\n
    '''
    store_psi_T_iter:bool
    sotre_intermideate_state:bool
    store_former_control_key:str

class Opt_result(Iter_info):
    def __init__(self,iter_stop:int,tlist_long:list,Hamiltonian,pulse_options:dict,options:Opt_result_options,runfolder=None,n_JT=1,JT_names=None,direction:bool=False,functional_info=None):
        super().__init__(iter_stop,runfolder,n_JT,JT_names,direction,functional_info)
        self.Hamiltonian = Hamiltonian
        self.pulse_options = pulse_options
        self.tlist_long = tlist_long
        self.initial_controls = None
        self.stored_controls = []
        self.psi_T = None
        self.psi_T_iter = []
        self.psi_t = None
        self.options = options

    def obtain_pulse(self):
        pulses = {}
        for Hi in self.Hamiltonian:
            if isinstance(Hi,list):
                pulses[Hi[1]] = self.pulse_options[Hi[1]]['args']["fit_func"]
        return pulses

    def store_psi_T(self,psi_T):
        self.psi_T = psi_T
        if self.options.store_psi_T_iter:
            self.psi_T_iter.append(psi_T)

    def store_initial_controls(self):
        initial_controls = {}
        for Hi in self.Hamiltonian:
            if isinstance(Hi,list):
                initial_controls[Hi[1]] = self.pulse_options[Hi[1]]['args']['fit_func'](self.tlist_long)
        self.initial_controls = initial_controls

    def obtain_pulse_real_sequence(self):
        pulses = []
        for Hi in self.Hamiltonian:
            if isinstance(Hi,list):
                pulses.append([self.pulse_options[Hi[1]]['args']["fit_func"](self.tlist_long),self.pulse_options[Hi[1]]['oct_lambda_a']])
        return pulses

    def write_pulse(self):
        pulses = self.obtain_pulse_real_sequence()
        for i in range(len(pulses)):
            if pulses[i][1]: # If oct_lambda_a == 0, do not print pulse.
                control_text = read_write.control2text(self.tlist_long,pulses[i][0])
                with open(self.runfolder+f'pulse_oct_{i}.dat','w') as pulse_f:
                    pulse_f.write(control_text)
    def write_psi_T(self):
        if len(self.psi_T) == 1:
            state_text = read_write.state2text(self.psi_T[0])
            with open(self.runfolder+'psi_final_after_oct.dat','w') as state_f:
                state_f.write(state_text)
        else:
            for i in range(len(self.psi_T)):
                state_text = read_write.state2text(self.psi_T[i])
                with open(self.runfolder+f'psi_{i}_final_after_oct.dat','w') as state_f:
                    state_f.write(state_text)

    def store_control(self,new_controls):
        '''
        storkey_key == 'last': only store the last two control scheme updated
                    == 'last': store all updated control scheme
        '''
        if self.options.store_former_control_key == 'all':
            if len(self.stored_controls) == 0:
                controls_0 = self.obtain_pulse()
                for H_i in self.Hamiltonian:
                    if isinstance(H_i,list):
                        controls_0[H_i[1]] = controls_0[H_i[1]](self.tlist_long)
                self.stored_controls.append(controls_0)
            self.stored_controls.append(new_controls)
        if self.options.store_former_control_key == 'last':
            if len(self.stored_controls) < 2:
                self.stored_controls.append(new_controls)
            else:self.stored_controls = [self.stored_controls[-1],new_controls]
        for H_i in self.Hamiltonian:
            if isinstance(H_i,list):
                if H_i[1] in new_controls.keys():
                    self.pulse_options[H_i[1]]['args']["fit_func"] =  interp1d(
                    self.tlist_long, new_controls[H_i[1]], kind="cubic", fill_value="extrapolate")
        if self.runfolder:
            self.write_pulse()
            self.write_psi_T()

    def plot_sotred_pulses(self,fig_name=None):
        colors = ['r','g','b','c','m','y']
        if self.initial_controls:
            self.stored_controls = [self.initial_controls] + self.stored_controls            
        alphas = np.linspace(0.5,1,len(self.stored_controls))
        for i in range(len(self.stored_controls)):
            pulse_count = 0
            for H_i in self.Hamiltonian:
                if isinstance(H_i,list):
                    plt.plot(self.tlist_long,self.stored_controls[i][H_i[1]],color=colors[pulse_count % len(colors)],alpha=alphas[i])
                    pulse_count += 1
        if len(self.stored_controls):
            if fig_name:
                if self.runfolder:fig_name = self.runfolder + fig_name
                plt.savefig(fig_name)
                plt.clf()
            else:
                plt.show()
    
    def plot_JT_iter(self,fig_name=None):
        for i in range(self.n_JT):
            plt.plot([self.JT_iter[iters][i] for iters in range(len(self.JT_iter))],label = self.JT_names[i])
        plt.legend(loc='best')
        plt.xlabel('iter')
        plt.ylabel('JT')
        if fig_name:
            if self.runfolder:fig_name = self.runfolder + fig_name
            plt.savefig(fig_name)
            plt.clf()
        else:
            plt.show()

class Monitor(Opt_result):
    #def __init__(self,iter_stop,tlist_long,Hamiltonian,pulse_options,options:Opt_result_options,runfolder,n_JT,JT_names,direction,func,x0,iter_stop,runfolder,n_JT,JT_name):
    def __init__(self,iter_stop,tlist,tlist_long,Hamiltonian,pulse_options,options:Opt_result_options,runfolder,n_JT,JT_names,functional_info,func,x0,approx_grad=False,order = 2):
        super().__init__(iter_stop,tlist_long,Hamiltonian,pulse_options,options,runfolder,n_JT,JT_names,0,functional_info)
        self.iter_stop = iter_stop
        self.last_control = x0
        self.tlist = tlist
        self.func = func
        self.iters = 0
        path_Path = Path(runfolder)
        path_Path.mkdir(exist_ok=True,parents=True)
        self.tlist_long = tlist_long
        self.approx_grad = approx_grad
        self.JT_new = None
        self.t_log = [time.time()]
        self.order = order
        self.__initial_run()

    def set_iter_params(self,factr:float,maxls:int,pgtol:float):
        '''
        parameters for l-bfgs-b:\n

        factr: float
        maxls:int
        pgtol:float
        '''
        self.factr = factr
        self.maxls = maxls
        self.maxfun = self.maxls * self.iter_stop + 1
        self.pgtol = pgtol

    def __initial_run(self):
        self.JT_new,psi_T = self.func(self.last_control,True,self.order)
        self.psi_T = psi_T
        self.t_log.append(time.time())
        dt = self.t_log[-1] - self.t_log[-2]
        #self.iter_log.log_iter_info(self.iters,self.JT_new,dt,ga_int=0)
        self.log_iter_info(self.iters,self.JT_new,dt,ga_int=0)
        self.iters += 1

    def cost_function(self,x):
        if self.approx_grad:self.JT_new,psi_T = self.func(x,self.approx_grad,self.order)
        else:self.JT_new,grad,psi_T = self.func(x,self.approx_grad,self.order)
        self.psi_T = psi_T
        if isinstance(self.JT_new,list):JT = self.JT_new[0]
        else:JT = self.JT_new
        if self.approx_grad:return JT
        else:return JT,grad

    def callback(self,x):
        ga_int = sum(np.abs(x-self.last_control)) * (self.tlist_long[1] - self.tlist_long[0])
        self.last_control = x
        self.t_log.append(time.time())
        dt = self.t_log[-1] - self.t_log[-2]
        #self.iter_log.log_iter_info(self.iters,self.JT_new,dt,JT_last=self.JT_iter[-1],ga_int=ga_int)
        self.log_iter_info(self.iters,self.JT_new,dt,JT_last=self.JT_iter[-1][0],ga_int=ga_int)
        new_controls = localTools.array2control(x,self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
        self.store_control(new_controls)
        self.iters += 1
    
    def __exit_clause(self,d):
        if d['warnflag'] == 0: return f'converged\n\t{d['task']}'
        if d['warnflag'] == 1: return 'too many function evaluations or too many iterations'
        if d['warnflag'] == 2: return f'stopped for another reason: \n\t{d['task']}'

    def log_finish_info(self,x,f,d):
        new_controls = localTools.array2control(x,self.tlist,self.pulse_options,self.Hamiltonian,self.tlist_long)
        self.store_control(new_controls)
        with open(self.runfolder+'finish_report.dat','w') as report_f:
            report_f.write(f'''L-BFGS-B optimization parameters: \n\tmax_iter {self.iter_stop}, \n\tJT_conv {self.factr:.2e} * eps, 
                           \n\tmax_num of line search {self.maxls}\n\tapprox_grad: {self.approx_grad}\n\tpgtol: {self.pgtol:.2e}\n''')
            report_f.write(f'Optimization finishes with functional value {f}\n')
            report_f.write(f'Exit Clause: {d['warnflag']} - {self.__exit_clause(d)}\n')
            report_f.write(f'Number of functional calls: {d['funcalls']}\n')
            report_f.write(f'Number of iterations: {d['nit']}\n')
        return 0
