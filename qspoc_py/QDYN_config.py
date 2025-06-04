from pathlib import Path
import time, qutip, qdyn, qdyn.model, qdyn.pulse, os, subprocess, numpy as np,platform,localTools,J_T_local,csv,read_write
import datetime

def addHam2model(num_qubit,Ham_num,tgrid,control_source,endTime,header,n_levels = False,dissipation=False,continue_from=False):
    ham_info = localTools.hamiltonian_info(num_qubit, Ham_num)
    num_correct_pulses = sum(ham_info) - ham_info[0]
    num_pulses = 0
    while num_pulses != num_correct_pulses:
        H = localTools.hamiltonian(num_qubit, Ham_num=Ham_num,n_levels = n_levels,dissipation = dissipation)
        if continue_from:
            control_args = localTools.control_generator_S2L(
                num_qubit, Ham_num, control_source, endTime, header)
        else:
            control_args = localTools.control_generator(
                num_qubit, Ham_num, control_source, endTime, header)
        model = qdyn.model.LevelModel()
        for i in range(len(H)):
            if isinstance(H[i],list):
                id_pulse_opts = pulse_options[H[i][1]]
                model.add_ham(
                    H[i][0],
                    pulse=qdyn.pulse.Pulse(
                        tgrid,
                        amplitude=H[i][1](tgrid, control_args[i - ham_info[0]]),
                        time_unit="iu",
                        ampl_unit="iu",
                        # is_complex="True",
                        config_attribs={
                            "filename": id_pulse_opts["filename"],
                            "oct_shape": id_pulse_opts["oct_shape"],
                            "t_rise": id_pulse_opts["t_rise"],
                            "t_fall": id_pulse_opts["t_rise"],
                            "oct_increase_factor": id_pulse_opts["oct_increase_factor"],
                            "oct_pulse_min": id_pulse_opts["oct_pulse_min"],
                            "oct_pulse_max": id_pulse_opts["oct_pulse_max"],
                            "oct_lambda_a": id_pulse_opts["oct_lambda_a"],
                            "oct_outfile": id_pulse_opts["oct_outfile"]}),
                        op_unit="iu")
            else:
                model.add_ham(H[i], op_unit="iu")
        num_pulses = len(model._pulse_ids)
    return model

def default_run_method():
    return {"immediate_return":True,"slurm":True,"store":True}

def runner(runfolder, jobname,runtime,mem,program_ID,n_cpu=1,run_method = None):
    if run_method:
        if 'immediate_return' in run_method:
            immediate_return = run_method["immediate_return"]
        else:immediate_return = True
        if 'slurm' in run_method:
            slurm = run_method["slurm"]
        else:slurm = True
        if 't_sleep' in run_method:
            t_sleep = run_method["t_sleep"]
        else:t_sleep = 10
    else:
        immediate_return = True
        slurm = True
        t_sleep = 10
    env = {**os.environ, "OMP_NUM_THREADS": str(n_cpu)}
    executable = ["inFidelity", "inFidelity_gate1", "inFidelity_gate2","prop"]
    localTools.regulate_pulses_in_folder(runfolder)
    os_system = platform.platform()
    if 'Linux' in os_system:
        if slurm:
            slrm_scrpt = localTools.text_content_local_run(jobname,n_cpu,runtime,mem,runner=f'./runner/{executable[program_ID]}',runfolder=runfolder)
            with open(f'{runfolder}/run_rf.sh','w') as f:
                f.write(slrm_scrpt)
            fortran_result = subprocess.run(["sbatch", f"{runfolder}/run_rf.sh"])
        else:
            fortran_result = subprocess.run(['./runner/'+executable[program_ID], runfolder], capture_output=True, text=True, env=env)       
        if immediate_return:
            time.sleep(0.1)
            return 0
        print(f'{runfolder} program starts at {datetime.datetime.now()}')
        while not any([os.path.exists(runfolder + 'psi_final_after_oct.dat'),os.path.exists(runfolder + 'psi0_final_after_oct.dat')]):
            time.sleep(t_sleep)
        if executable[program_ID] != "prop":
            iters, J_T = np.split(np.loadtxt(runfolder + "oct_iters.dat", usecols=(0, 1)), 2, axis=1)
            runner_result = J_T[-1][0]
        else:runner_result = 1
    else: runner_result = 'OS system is not Linux, pass'
    return runner_result

def mem_routine(num_qubit,n_states,t_step):
    if num_qubit == 4:return 70
    mem_1q = 4e-3 # here looks suspicious
    mem_2q = 1e-2
    if num_qubit == 1 :return int(n_states * mem_1q * t_step * 1.1) + 2 * n_states
    if num_qubit == 2 :return int(n_states * mem_2q * t_step * 1.1) + 2 * n_states

def runtime_routine():
    return '2-12:00:00'

def rotate_state_call(target_state,num_qubit,T,direction = 1):
    return qutip.Qobj(
        localTools.rotate_state(target_state, num_qubit, direction, T))

def rotate_matrix_call(lab_mat,num_qubit,T,direction = 1):
    return qutip.Qobj(
        localTools.rotate_matrix(lab_mat, num_qubit, direction, T))

def qdyn_prop(
    Ham_num,  # Hamiltonian
    qdyn_tlist,  # tuple of the form (T, Nt)
    initial_state,
    runfolder,
    control_source,
    header=None,
    num_qubit=3,
    dissipation = False,
    **user_kwargs,
):
    runfolder=Path(runfolder)
    runfolder.mkdir(parents=True, exist_ok=True)
    dt = (qdyn_tlist[0]) / (qdyn_tlist[1] - 1)
    tgrid = np.linspace(
        float(dt / 2),
        float(qdyn_tlist[0] - dt / 2),
        qdyn_tlist[1] - 1,
        dtype=np.float64,
    )  #! here the default is np.float 64, it has been changed manually, also in QDYN python package
    if "control" in user_kwargs:
        user_kwargs.pop("control")
    if isinstance(initial_state, str):
        initial_state = localTools.canoGHZGen(num_qubit, initial_state)
    model = addHam2model(num_qubit,Ham_num,tgrid,control_source,qdyn_tlist[0],header,1,1,2,dissipation = dissipation)
    print('Ham added to model')
    model.set_propagation(
        qdyn_tlist[0], qdyn_tlist[1], time_unit="iu", prop_method="cheby")
    model.add_state(initial_state, "initial")
    user_variables = {}
    if not os.path.exists(runfolder):
        os.mkdir(runfolder)
    user_variables["runfolder"] = str(Path(runfolder))
    user_variables.update(user_kwargs)
    model.user_data = user_variables
    model.write_to_runfolder(str(runfolder))  # write everything to runfolder
    mem = mem_routine(num_qubit,1,qdyn_tlist[1])
    runner(str(runfolder)+'/','prop','00:10:00',mem,program_ID=5,n_cpu=1,run_method = {"immediate_return":False,"slurm":False})
    return model

def qdyn_model(
    Ham_num,  # Hamiltonian
    qdyn_tlist,  # tuple of the form (T, Nt)
    initial_states,
    target_states,
    runfolder,
    control_source,
    header=None,
    num_qubit=3,
    lambda_a=10,
    iter_stop=100,
    JT=0,
    t_rise=1,
    dissipation = False,
    run_method = False,
    **user_kwargs,
):
    if not run_method:run_method=default_run_method()
    n_states = len(initial_states)
    n_levels = int(len(target_states[0]) ** (1/num_qubit))
    if type(initial_states[0]) != qutip.Qobj:
        for i in range(n_states):
            initial_states[i] = qutip.Qobj(initial_states[i])
            target_states[i] = qutip.Qobj(target_states[i])
    target_states = rotate_state_call(target_states,Ham_num,num_qubit,qdyn_tlist[0])
    dt = (qdyn_tlist[0]) / (qdyn_tlist[1] - 1)
    tgrid = np.linspace(
        float(dt / 2),
        float(qdyn_tlist[0] - dt / 2),
        qdyn_tlist[1] - 1,
        dtype=np.float64,
    )  #! here the default is np.float 64, it has been changed manually, also in QDYN python package
    # Initialize model
    model = addHam2model(num_qubit,Ham_num,tgrid,control_source,qdyn_tlist[0],header,t_rise,lambda_a,n_levels,dissipation = dissipation)
    obj_path = Path(runfolder)
    obj_path.mkdir(parents=True, exist_ok=True)
    model.set_propagation(
        qdyn_tlist[0], qdyn_tlist[1], time_unit="iu", prop_method="cheby")
    if JT:
        J_T_conv = 9.8e-5
        delta_J_T_conv = 1e-10
    else:
        J_T_conv = 9.8e-3
        delta_J_T_conv = 1e-5
    mem = mem_routine(num_qubit,n_states,qdyn_tlist[1])
    model.set_oct(
        method="krotovpk",
        max_ram_mb=mem,
        J_T_conv=J_T_conv,
        delta_J_T_conv=delta_J_T_conv,
        iter_dat="oct_iters.dat",
        continue_=False,
        params_file="oct_params.dat",
        limit_pulses=True,
        # keep_pulses="prev",
        iter_stop=iter_stop)
    for i in range(n_states):
        model.add_state(initial_states[i], "initial")
        model.add_state(target_states[i], "final")
    user_variables = {}
    user_variables["runfolder"] = str(Path(runfolder))
    user_variables.update(user_kwargs)
    model.user_data = user_variables
    model.write_to_runfolder(str(runfolder))
    jobname = runfolder.split('/')[1]
    runtime = runtime_routine()
    program_ID = int(np.log2(n_states))
    runner_result = runner(runfolder,jobname,runtime,mem = mem + 10,program_ID=program_ID,run_method = run_method)
    return runner_result

