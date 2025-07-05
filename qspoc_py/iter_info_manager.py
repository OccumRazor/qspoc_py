class Iter_info:
    def __init__(self,n_JT,iter_stop,runfolder=None):
        self.n_JT = n_JT
        self.iter_stop = iter_stop
        self.runfolder = runfolder
        self.iter_str_len = len(str(iter_stop)) + 2
        return 0
    
    