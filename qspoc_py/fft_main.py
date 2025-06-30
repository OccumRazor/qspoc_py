from . import read_write
from scipy.fft import fft,fftfreq,irfft,fftshift
import matplotlib.pyplot as plt,numpy as np
from scipy.interpolate import interp1d
from scipy import signal

def fit_filtered_control(tlist,filtered_amplitude):
    if tlist[0] == 0: 
        t_start = 0
        t_end = tlist[-1]
    else:
        half_dt = (tlist[1] - tlist[0])/2
        t_start = tlist[0] - half_dt
        t_end = tlist[-1] + half_dt
    control_fit = interp1d(np.linspace(t_start,t_end,len(filtered_amplitude)),filtered_amplitude, kind="cubic", fill_value="extrapolate")
    return control_fit(tlist)

def control_fft(tlist, amplitudes,fig_name=None):
    """
    Applies a window function in the frequency domain to a time-domain signal.

    Args:
        tlist (np.ndarray): A 1-D numpy array representing the time points of the signal.
        amplitude (np.ndarray): A 1-D numpy array representing the amplitude of the signal at each time point.
        window (np.ndarray): A 1-D numpy array representing the window function to apply in the frequency domain.
                             It should have the same length as the FFT of the amplitude.

    Returns:
        np.ndarray: A 1-D numpy array representing the time-domain signal after the window has been applied in the frequency domain.
    """
    # 1. Perform the Fast Fourier Transform (FFT)
    #fit = interp1d(tlist,amplitude, kind="cubic", fill_value="extrapolate")
    #tlist = np.linspace(0,1,10000)
    #amplitude = fit(tlist)
    #fft_result = np.fft.fft(amplitude)
    #frequencies = np.fft.fftfreq(amplitude.size, d=tlist[1] - tlist[0])
    fft_results = []
    for amplitude in amplitudes:
        if amplitude[1]:
            fft_results.append(fft(amplitude[0]))
    frequencies = fftfreq(amplitudes[0][0].size, d=tlist[1] - tlist[0])

    plt.figure(figsize=(10, 5))

    plt.subplot(1,2, 1)
    for i in range(len(amplitudes)):
        if amplitudes[i][1]:
            plt.plot(tlist, amplitudes[i][0],label = rf'$\epsilon_{i}$')
    plt.title('FFT Amplitude')
    plt.xlabel('t')
    plt.ylabel('Control')
    plt.legend(loc='best')
    plt.subplot(1,2, 2)
    for i in range(len(fft_results)):
        plt.plot(frequencies, np.abs(fft_results[i]),'--',label = rf'$\epsilon_{i}$')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend(loc='best')
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name)
        plt.clf()
    else:
        plt.show()

def gen_window(n_fft,threshold):
    window = np.zeros(n_fft)
    full_bin = int(0.5 * (n_fft * threshold + 1)) 
    window[:full_bin] = 1.0
    end_bin = int(n_fft * threshold + 1)
    for loc in range(end_bin - full_bin):
        window[loc + full_bin] = np.sin(-0.5 * np.pi -0.5 * np.pi * loc / (end_bin - full_bin)) ** 2
    return window

def fft_filter(tlist, amplitude,threshold):
    n_fft = amplitude.size
    window = gen_window(n_fft,threshold)
    fft_result = fft(amplitude)
    frequencies = fftfreq(amplitude.size, d=(tlist[1] - tlist[0]))
    max_freq = max(frequencies)
    if max_freq > 10 ** 9:
        freq_unit = 'GHZ'
        frequencies /= 10 ** 9
    elif max_freq > 10 ** 6:
        freq_unit = 'MHZ'
        frequencies /= 10 ** 6
    elif max_freq > 10 ** 3:
        freq_unit = 'kHZ'
        frequencies /= 10 ** 3
    else:freq_unit = 'HZ'
    #frequencies = fftshift(frequencies)
    # 2. Apply the window function in the frequency domain
    # Ensure the window has the same length as the FFT result
    if len(window) != len(fft_result):
        raise ValueError("The window length must be equal to the FFT result length.")
    filtered_fft = fft_result * window
    # 3. Perform the Inverse Fast Fourier Transform (IFFT) to return to the time domain
    #filtered_amplitude = 2 * np.fft.ifft(filtered_fft)  # Take the real part as the result
    filtered_amplitude = 2 * irfft(filtered_fft)  # Take the real part as the result
    return fit_filtered_control(tlist,filtered_amplitude)

def apply_window_fft(tlist, amplitude, window):
    """
    Applies a window function in the frequency domain to a time-domain signal.

    Args:
        tlist (np.ndarray): A 1-D numpy array representing the time points of the signal.
        amplitude (np.ndarray): A 1-D numpy array representing the amplitude of the signal at each time point.
        window (np.ndarray): A 1-D numpy array representing the window function to apply in the frequency domain.
                             It should have the same length as the FFT of the amplitude.

    Returns:
        np.ndarray: A 1-D numpy array representing the time-domain signal after the window has been applied in the frequency domain.
    """
    # 1. Perform the Fast Fourier Transform (FFT)
    #fit = interp1d(tlist,amplitude, kind="cubic", fill_value="extrapolate")
    #tlist = np.linspace(0,1,10000)
    #amplitude = fit(tlist)
    #fft_result = np.fft.fft(amplitude)
    #frequencies = np.fft.fftfreq(amplitude.size, d=tlist[1] - tlist[0])
    fft_result = fft(amplitude)
    frequencies = fftfreq(amplitude.size, d=(tlist[1] - tlist[0]))
    max_freq = max(frequencies)
    if max_freq > 10 ** 9:
        freq_unit = 'GHZ'
        frequencies /= 10 ** 9
    elif max_freq > 10 ** 6:
        freq_unit = 'MHZ'
        frequencies /= 10 ** 6
    elif max_freq > 10 ** 3:
        freq_unit = 'kHZ'
        frequencies /= 10 ** 3
    else:freq_unit = 'HZ'
    #frequencies = fftshift(frequencies)
    # 2. Apply the window function in the frequency domain
    # Ensure the window has the same length as the FFT result
    if len(window) != len(fft_result):
        raise ValueError("The window length must be equal to the FFT result length.")

    filtered_fft = fft_result * window
    # 3. Perform the Inverse Fast Fourier Transform (IFFT) to return to the time domain
    #filtered_amplitude = 2 * np.fft.ifft(filtered_fft)  # Take the real part as the result
    filtered_amplitude = 2 * irfft(filtered_fft)  # Take the real part as the result

    # 4. Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1,2, 1)
    plt.plot(frequencies, np.abs(fft_result)/max(np.abs(fft_result)),label = 'initial')
    #plt.plot( np.abs(fft_result)/max(np.abs(fft_result)),label = 'initial')

    plt.plot(frequencies, np.abs(filtered_fft)/max(np.abs(filtered_fft)),'--',label = 'filtered')
    #plt.plot(np.abs(filtered_fft)/max(np.abs(filtered_fft)),'--',label = 'filtered')
    plt.plot(frequencies,window,label='window')
    plt.title('FFT Amplitude')
    plt.xlabel(f'Frequency ({freq_unit})')
    plt.ylabel('Magnitude')
    #plt.plot(window,label='window')
    plt.legend(loc='best')
    plt.subplot(1,2, 2)
    plt.plot(tlist, amplitude,label = 'initial')
    plt.title('Time Domain Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.plot(np.linspace(tlist[0],tlist[-1],len(filtered_amplitude)), filtered_amplitude,'--',label='filtered')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return filtered_amplitude

if __name__ == '__main__':
    # Example Usage
    fs = 100  # Sampling frequency (Hz)
    t = np.linspace(0, 1, 1000, endpoint=False)  # Time vector
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)  # Example signal

    # Create a simple rectangular window in the frequency domain
    n_fft = signal.size
    center_freq_bin = int(n_fft * 0 / fs)  # Center around 10 Hz
    width_freq_bin = int(n_fft * 10 / fs)    # Width of 5 Hz on each side
    window_freq = np.zeros(n_fft)
    start_bin = max(0, center_freq_bin - width_freq_bin // 2)
    end_bin = min(n_fft, center_freq_bin + (width_freq_bin + 1) // 2)
    window_freq[start_bin:end_bin] = 1.0

    filtered_signal = apply_window_fft(t, signal, window_freq)


'''
# sample spacing
tlist,amp = read_write.controlReader('data/bnm_X_1muS/pulse_oct_0.dat')
n_fft = len(tlist)
n_fft=10000
fs = 100
center_freq_bin = int(n_fft * 0 / fs)  # Center around 10 Hz
width_freq_bin = int(n_fft * 5 / fs)    # Width of 5 Hz on each side
window_freq = np.zeros(n_fft)
start_bin = max(0, center_freq_bin - width_freq_bin // 2)
end_bin = min(n_fft, center_freq_bin + (width_freq_bin + 1) // 2)
window_freq[start_bin:end_bin] = 1.0
window = np.kaiser(n_fft,1)
apply_window_fft(tlist,amp,window_freq)
fit = interp1d(tlist, amp, kind="cubic", fill_value="extrapolate")
N = 10000
amp_L = fit(np.linspace(0,1,N))
sp = np.fft.fft(np.fft.fft(amp_L))
freq = np.fft.fftfreq(N)
import matplotlib.pyplot as plt
plt.plot(freq,sp.real,freq,sp.imag)
plt.grid()
plt.show()
'''