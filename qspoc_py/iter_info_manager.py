import time,numpy as np
from pathlib import Path

class Iter_info:
    def __init__(self,iter_stop,runfolder=None,n_JT=1,JT_names=None,direction=1):
        '''
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
        if not JT_names:
            JT_names = ['JT'] + [f'JT_{i}' for i in range(n_JT-1)]
        if self.runfolder:
            self.out_stream = open(runfolder + 'oct_iters.dat','w')
        message = f'#{' ' * (self.iter_str_len - 3)}iter{' ' * 4}'
        for JT_name in JT_names:
            message += JT_name + ' ' * (14 - len(JT_name))
        message += f'dJT{' ' * 11}ga_int{' ' * 8}dt'
        if self.runfolder:self.out_stream.write(message+'\n')
        else:print(message)
    
    def log_iter_info(self,iters,JT_new,dt,JT_last=None,ga_int=None):
        if not JT_last:
            dJT = 0.0
        else:
            dJT = JT_last - JT_new[0]
            if self.direction:
                dJT = -dJT
        if not ga_int:ga_int = 0.0
        message = f'{' ' * (self.iter_str_len-len(str(iters))+2)}{iters}{' ' * 4}'
        for i in range(self.n_JT): message += f'{JT_new[i]:.8f}{' ' * 4}'
        message += f'{dJT:.8f}{' ' * 4}{ga_int:.8f}{' ' * 4}{dt:.2f}'
        if self.runfolder:
            self.out_stream.write(message+'\n')
            self.out_stream.flush()
        else:print(message)
    
    def log_break_info(self,JT_new,JT_last,iters,ga_int,ga_bound):
        if self.direction:
            if JT_new[0] < JT_last:message = f'#{' ' * (self.iter_str_len - len(str(iters)))}{iters} monotonicity breaks, JT_new = {JT_new[0]}, increase lambda_a by a factor of 2.'
            else:message = f'# ga_int ({ga_int}) > ga_bound ({ga_bound}), increase lambda_a by a factor of 2.'
        else:
            if JT_new[0] > JT_last:message = f'#{' ' * (self.iter_str_len - len(str(iters)))}{iters} monotonicity breaks, JT_new = {JT_new[0]}, increase lambda_a by a factor of 2.'
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


class Monitor:
    def __init__(self,func,x0,iter_stop,runfolder,n_JT,JT_name):
        self.last_control = x0
        self.func = func
        self.JT_iter = []
        self.iters = 0
        self.t_log = [0]
        path_Path = Path(runfolder)
        path_Path.mkdir(exist_ok=True,parents=True)
        self.iter_log = Iter_info(iter_stop,runfolder,n_JT,JT_name,0)

    def time_stamp(self):
        self.t_log.append(time.time())

    def cost_function(self,x):
        JT_new,grad = self.func(x)
        if self.iters:dt = self.t_log[-1] - self.t_log[-2]
        else:dt = 0
        ga_int = sum(np.abs(x-self.last_control))
        if len(self.JT_iter):self.iter_log.log_iter_info(self.iters,JT_new,dt,JT_last=self.JT_iter[-1],ga_int=ga_int)
        else:self.iter_log.log_iter_info(self.iters,JT_new,dt,ga_int=ga_int)
        self.JT_iter.append(JT_new[0])
        self.iters += 1
        if isinstance(JT_new,list):return JT_new[0],grad
        else:return JT_new,grad

    def callback(self,x):
        self.t_log.append(time.time())