import random, os, numpy as np,time,scipy
from scipy.interpolate import interp1d
from . import read_write

def del_redundent(folder,black_list):
    files = os.listdir(folder)
    for file in files:
        if any([key_word in file for key_word in black_list]):
            #print(f'existence of {folder+file}:{os.path.exists(folder+file)}')
            time.sleep(0.2)
            os.remove(folder+file)
    return 0

def read_oct_iters(working_folder,delta_JT = False):
    if not os.path.exists(working_folder+'/oct_iters.dat'):
        if delta_JT: return [1,0]
        return 1
    this_JTS = np.loadtxt(working_folder+'/oct_iters.dat', usecols=(1))
    if delta_JT:return [this_JTS[-1],(this_JTS[-2]-this_JTS[-1])/this_JTS[-1]]
    return this_JTS[-1]

def sin_sq(t,T,t_rise,rise):
    omega = np.pi / 2 / t_rise
    if rise:
        return np.sin(omega * t) ** 2
    else:
        return np.sin(omega * (T - t)) ** 2

def flattop_pulse(T,t_rise,t_fall):
    def calculate_output(t_val):
        if isinstance(t_val, np.ndarray):
            mask_rise_region = t_val < t_rise
            mask_fall_region = t_val > (T - t_fall)
            mask_constant_region = ~(mask_rise_region | mask_fall_region)
            output_array = np.empty_like(t_val, dtype=t_val.dtype if t_val.dtype == float else np.float64) # Ensure float output for sin
            if np.any(mask_rise_region):
                output_array[mask_rise_region] = sin_sq(t_val[mask_rise_region], T, t_rise, True)
            if np.any(mask_fall_region):
                output_array[mask_fall_region] = sin_sq(t_val[mask_fall_region], T, t_fall, False)
            if np.any(mask_constant_region):
                output_array[mask_constant_region] = np.full(
                    np.sum(mask_constant_region),
                    1,
                    dtype=output_array.dtype)
            return output_array
        elif isinstance(t_val, (int, float)):
            if t_rise < t_val < T - t_fall:
                return type(t_val)(1)
            elif t_val < t_rise:
                return sin_sq(t_val,T,t_rise,True)
            else:
                return sin_sq(t_val,T,t_fall,False)
        else:
            # Fallback for other types, return a generic constant or raise an error
            # For this problem, we assume t is numeric (int, float, or np.ndarray)
            raise TypeError("Unsupported type for t. Must be a number or a numpy array.")
    return lambda t:calculate_output(t)
    #return {'fit_func':lambda t:calculate_output(t)}

def flattop(t, t_start, t_stop, t_rise, t_fall):
    func_obj = flattop_pulse(t_stop,t_rise,t_fall)
    return func_obj(t)

'''
def flattop(t, t_start, t_stop, t_rise, t_fall):
    if t_start <= t <= t_stop:
        f = 1.0
        if t <= t_start + t_rise:
            f = blackman(t, t_start, t_start + 2 * t_rise)
        elif t >= t_stop - t_fall:
            f = blackman(t, t_stop - 2 * t_fall, t_stop)
        return f
    else:
        return 0.0
'''

def box(t, t_start, t_stop):
    if t < t_start:
        return 0.0
    if t > t_stop:
        return 0.0
    return 1.0

def blackman(t, t_start, t_stop, a=0.16):
    T = t_stop - t_start
    box_vec = np.vectorize(box)
    return (
        0.5
        * box_vec(t, t_start, t_stop)
        * (
            1.0
            - a
            - np.cos(2.0 * np.pi * (t - t_start) / T)
            + a * np.cos(4.0 * np.pi * (t - t_start) / T)
        )
    )

def S(t,t_start,t_stop,t_rise,t_fall):
    return flattop(t, t_start=t_start, t_stop=t_stop, t_rise=t_rise, t_fall=t_fall)

def half_step_tlist(qdyn_tlist):
    if len(qdyn_tlist) == 3:qdyn_tlist=qdyn_tlist[1:]
    dt = (qdyn_tlist[0]) / (qdyn_tlist[1] - 1)
    tgrid = np.linspace(
        float(dt / 2),
        float(qdyn_tlist[0] - dt / 2),
        qdyn_tlist[1] - 1,
        dtype=float,
    )  #! here the default is np.float 64, it has been changed manually, also in QDYN python package
    return tgrid

def control_generator_S2L(num_qubit, control_source, endTime=None, header=None):
    control_args = []
    for i in range(num_qubit):
        if os.path.isfile(control_source + f"{header}_{i}.dat"):
            detupleTlist, detupleGuess = read_write.controlReader(
                control_source + f"{header}_{i}.dat"
            )
            oriTime = float(detupleTlist[-1] + (detupleTlist[-1] - detupleTlist[-2])/2)
        else:
            raise KeyError(f"Unable to find control in {control_source}{header}, {header}_{i}.dat does not exist.")
        former_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        new_tlist = half_step_tlist([endTime, len(detupleGuess) + 1])
        delta_T = endTime - oriTime
        new_Amp = []
        for t in new_tlist:
            if t<= delta_T: new_Amp.append(0)
            else: new_Amp.append(former_fit(float(t-delta_T)))
        cubicSpline_fit = interp1d(
            new_tlist, new_Amp, kind="cubic", fill_value="extrapolate")
        control_args.append({"fit_func": cubicSpline_fit})
    return control_args

def control_generator(
    n_controls, control_source, endTime=None, header=None
):
    if isinstance(control_source, str):
        control_args = control_generator_read(
            n_controls, control_source, header, endTime)
    else:
        control_args = control_generator_random(
            n_controls, control_source, endTime)
    return control_args

def control_generator_read(n_controls, control_source, header, endTime):
    control_args = []
    for i in range(n_controls):
        if os.path.isfile(control_source + f"{header}_{i}.dat"):
            detupleTlist, detupleGuess = read_write.controlReader(
                control_source + f"{header}_{i}.dat"
            )
            detupleTlist = half_step_tlist([endTime, len(detupleGuess) + 1])
        else:
            raise KeyError(f"Unable to find control in {control_source}{header}, {header}_{i}.dat does not exist.")
        cubicSpline_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        control_args.append({"fit_func": cubicSpline_fit})
    return control_args


def abs_wrapper(original_func):
    def new_func(t):
        return np.abs(original_func(t))
    return new_func

def control_generator_random(n_controls, guess_amps, endTime,positive = False):
    n_freq = 10
    control_args = []
    for i in range(n_controls):
        if isinstance(guess_amps,list):guess_amp = guess_amps[i]
        else:guess_amp = guess_amps
        amps = [guess_amp/n_freq * random.random() for _ in range(n_freq)]
        freqs = [2*np.pi*(k+1)/endTime for k in range(n_freq)]
        detupleTlist = np.linspace(0, endTime, 1001)
        detupleGuess = amps[0] * np.sin(freqs[0]*detupleTlist)
        for j in range(1,n_freq):
            detupleGuess += amps[j] * np.sin(freqs[j]*detupleTlist)
        cubicSpline_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        if positive: 
            control_args.append({"fit_func": abs_wrapper(cubicSpline_fit)})
        else:
            control_args.append({"fit_func": cubicSpline_fit})
    return control_args

'''
def control_generator_random(n_controls, guess_amps, endTime,positive = False):
    num_points = 15
    control_args = []
    for i in range(n_controls):
        if isinstance(guess_amps,list):guess_amp = guess_amps[i]
        else:guess_amp = guess_amps
        detupleGuess = (
            [0,0] + [guess_amp * random.random() - guess_amp / 2 for _ in range(num_points)] + [0,0])
        detupleTlist = np.linspace(0, endTime, len(detupleGuess))
        cubicSpline_fit = interp1d(
            detupleTlist, detupleGuess, kind="cubic", fill_value="extrapolate"
        )
        if positive: 
            control_args.append({"fit_func": abs_wrapper(cubicSpline_fit)})
        else:
            control_args.append({"fit_func": cubicSpline_fit})
    return control_args
'''

sq_dict = {
    "I": [[1, 0], [0, 1]],
    "X": [[0, 1], [1, 0]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
    "0": [[1, 0], [0, 0]],
    "1": [[0, 0], [0, 1]],
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
}


def fastKron(iptString):
    # Wenn es in iptString '+' gibt, zuerst iptString.split('+'), dann alle sammeln.
    if not isinstance(iptString, str):
        raise TypeError(
            "Input must be a string (consisted of +, 0, 1, I, H, X, Y and Z)"
        )
    if len(iptString) == 0:
        return np.array([[1]])
    if "+" in iptString:
        addMats = iptString.split("+")
        addMats = [fastKron(addMat) for addMat in addMats]
        return sum(addMats)
    kronMat = sq_dict[iptString[0]]
    for thisKey in iptString[1:]:
        kronMat = np.kron(kronMat, sq_dict[thisKey])
    return np.array(kronMat)

def isinstanceVector(state):
    # this function check whether the state is a ket/bra or not.
    if not isinstance(state, list) and not isinstance(state, np.ndarray):
        return 1
    if len(state[0]) == 1:
        return 1
    return 0

def densityMatrix(state):
    if not isinstanceVector(state):
        return state
    if isinstance(state[0], list) or isinstance(state[0], np.ndarray):
        state = list(np.array(state).T[0])
    dm = np.zeros([len(state),len(state)],dtype=np.complex128)
    for i in range(len(state)):
        for j in range(len(state)):
            dm[i][j] = complex(state[i] * np.conjugate(state[j]))
    return np.array(dm)


def rotate_state(state, H0, direction=0):
    if direction:
        return np.matmul(
            scipy.linalg.expm(1j * H0), state
        )  # \psi_RWA=U\dagger\psi_nR, return RWA frame state.
    else:
        return np.matmul(
            scipy.linalg.expm(-1j * H0), state
        )  # \psi_nR=U\psi_RWA, return lab frame state.

def rotate_matrix(mat, H0, direction=0):
    if direction:return np.matmul(np.matmul(scipy.linalg.expm(1j*H0),mat),scipy.linalg.expm(-1j*H0))
    else:return np.matmul(np.matmul(scipy.linalg.expm(-1j*H0),mat),scipy.linalg.expm(1j*H0))

def random_guess(t, control_args):
    fit_func = control_args.get("fit_func")
    return fit_func(t)

def random_guess_cos(t, control_args):
    fit_func = control_args.get("fit_func")
    frequency = control_args.get("freqX")
    return fit_func(t) * np.cos(frequency * t) * 2

def Hamiltonian(num_qubit=3, **kwargs):
    H0, Hc = Hamiltonian_Spin_Chain(num_qubit, **kwargs)
    return [qutip.Qobj(H0)] + [[qutip.Qobj(Hc[i]),lambda t,args:random_guess(t,args)] for i in range(len(Hc))]


