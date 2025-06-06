from . import read_write
from scipy.fft import fft,fftfreq,irfft
import matplotlib.pyplot as plt,numpy as np
from scipy.interpolate import interp1d

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
    frequencies = fftfreq(amplitude.size, d=tlist[1] - tlist[0])

    # 2. Apply the window function in the frequency domain
    # Ensure the window has the same length as the FFT result
    if len(window) != len(fft_result):
        raise ValueError("The window length must be equal to the FFT result length.")

    filtered_fft = fft_result * window

    # 3. Perform the Inverse Fast Fourier Transform (IFFT) to return to the time domain
    #filtered_amplitude = np.fft.ifft(filtered_fft)  # Take the real part as the result
    filtered_amplitude = irfft(filtered_fft)  # Take the real part as the result

    # 4. Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1,2, 1)
    plt.plot(frequencies, np.abs(fft_result),label = 'initial')
    plt.title('FFT Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.plot(frequencies, np.abs(filtered_fft),'--',label = 'filtered')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(window)
    plt.legend(loc='best')
    plt.subplot(1,2, 2)
    plt.plot(tlist, amplitude,label = 'initial')
    plt.title('Time Domain Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    #plt.plot(tlist, filtered_amplitude,'--',label='filtered')
    plt.plot(filtered_amplitude,'--',label='filtered')
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