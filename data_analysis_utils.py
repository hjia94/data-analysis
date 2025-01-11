'''
General unitility functions used for all data analysis.

Arthur: Jia Han
Date: 2024-02-28

Included functions:
- get_files_in_folder
- get_number_before_keyword
- save_to_npy
- read_from_npy
'''

import os
import sys
import re
import numpy as np
from scipy import signal
import scipy.constants as const
import matplotlib.pyplot as plt

from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional
from numpy.typing import NDArray

sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\Nov-2024")
from read_scope_data import read_trc_data
from plot_utils import plot_original_and_baseline, plot_subtracted_signal
#===========================================================================================================
#===========================================================================================================

def get_files_in_folder(folder_path, modified_date=None, omit_keyword=None):
    """
    Get a list of all files in a given folder and its subfolders.

    Args:
        folder_path (str): The path to the folder.
        modified_date (str, optional): The specific date in the format 'YYYY-MM-DD'. Defaults to None.
        omit_keyword (str, optional): The keyword to omit files. Defaults to None.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if omit_keyword is not None and omit_keyword in file:
                continue
            if modified_date is None:
                file_list.append(file_path)
            else:
                last_modified = os.path.getmtime(file_path)
                last_modified_date = datetime.fromtimestamp(last_modified).date()
                if last_modified_date == datetime.strptime(modified_date, '%Y-%m-%d').date():
                    file_list.append(file_path)
    return file_list
    
def get_number_before_keyword(string, keyword, verbose=False):
    """
    Extracts the number before a given keyword in a string.

    Parameters:
    string (str): The input string.
    keyword (str): The keyword to search for.
    verbose (bool, optional): If True, prints the number found. Defaults to False.

    Returns:
    float or None: The number found before the keyword, or None if no match is found.
    """
    match = re.search(r'(\d+)'+keyword, string)
    if match:
        if verbose:
            print("Number found:", match.group(1))
        return float(match.group(1))
    else:
        if verbose:
            print("No match found.")
        return None
    
def save_to_npy(data, npy_file_path):
    """
    Save data to a .npy file.

    Parameters:
    data (numpy.ndarray): The data to be saved.
    npy_file_path (str): The file path to save the .npy file.

    Returns:
    None
    """
    if os.path.exists(npy_file_path):
        user_input = input("The file already exists. Do you want to continue and overwrite it? (y/n): ")
        if user_input.lower() == 'n':
            return None
        if user_input.lower() != 'y' and user_input.lower() != 'n':
            print('Not the correct response. Please type "y" or "n"')

    np.save(npy_file_path, data)
    print(f"Data saved to {npy_file_path}")

def read_from_npy(npy_file_path):
    """
    Read data from a NumPy file (.npy).

    Args:
        npy_file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray or None: The loaded data if the file exists, None otherwise.
    """
    if not os.path.exists(npy_file_path):
        print(f"The file {npy_file_path} does not exist.")
        return None
    data = np.load(npy_file_path, allow_pickle=True)
    return data

#===============================================================================================================================================

def ion_sound_speed(Te, Ti, mi=const.m_p):
    '''
    Compute ion sound speed in m/s
    input:
    Te: electron temperature in eV
    Ti: ion temperature in eV
    mi: ion mass in kg
    '''
    gamma = 5/3 # adiabatic index; monoatomic gas is 5/3
    cs = np.sqrt((const.e * (Te + gamma*Ti)) / (mi))

    return cs

#===============================================================================================================================================
'''
PhotonPulse and Photons classes pulse-width analysis 
Original code from Pat written for McPhereson spectrometer data.
Performs pulse detection and analysis on photon pulses.
Can be used for data obtained from any PMT in general.
The PhotonPulse and Photons classes are used for analyzing photon pulses in time series data, typically from PMT signals.

Example usage:
    times = np.array([...])
    signal = np.array([...])

    detector = Photons(times, signal)
    detector.reduce_pulses()

    # Get results
    pulse_times, pulse_areas = detector.get_pulse_arrays()
    print(f"Detected {detector.pulse_count} pulses")

    # Access individual pulses
    for pulse in detector.pulses:
        print(f"Pulse at {pulse.time}ms with area {pulse.area}")
'''
@dataclass
class PhotonPulse:
    """Represents a single photon pulse detection."""
    time: float  # Average time of pulse (ms)
    area: float  # Integrated area of pulse
    width: float # Temporal width of pulse (ms)

class Photons:
    """Analyzes photon pulses in time series data.
    
    Attributes:
        offset (float): Baseline offset of the signal
        std_dev (float): Standard deviation of the baseline
        dt (float): Time step between samples
        threshold (float): Detection threshold for pulses
        pulses (list[PhotonPulse]): Detected photon pulses
    """
    
    def __init__(self, 
                 times: NDArray[np.float64], 
                 signal: NDArray[np.float64], 
                 threshold_multiplier: float = 7.0,
                 cutoff_freq: float = 0.01,
                 negative_pulses: bool = False):
        """Initialize photon pulse detector.
        
        Args:
            times: Time array in milliseconds
            signal: Signal amplitude array
            threshold_multiplier: Number of standard deviations above baseline for detection
            cutoff_freq: Cutoff frequency for the low-pass filter as a fraction of the Nyquist frequency
            negative_pulses: If True, detect negative-going pulses instead of positive
        
        Raises:
            ValueError: If input arrays have different lengths or are empty
        """
        if len(times) != len(signal):
            raise ValueError("Time and signal arrays must have same length")
        if len(times) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        self.times = times
        self.signal = signal
        self.cutoff_freq = cutoff_freq
        self._compute_signal_properties()
        self._detect_pulses(threshold_multiplier, negative_pulses)
        
    def _compute_signal_properties(self) -> None:
        """Compute baseline properties of the signal using a low-pass filter."""
        # Design a low-pass Butterworth filter
        b, a = signal.butter(2, self.cutoff_freq, btype='low')
        
        # Apply the filter to get the baseline
        self.baseline = signal.filtfilt(b, a, self.signal)
        
        # Subtract baseline from signal to get residuals
        residuals = self.signal - self.baseline
        
        # Compute standard deviation from residuals
        self.std_dev = np.std(residuals)
        self.dt = np.mean(np.diff(self.times))
        
    def _detect_pulses(self, threshold_multiplier: float, negative_pulses: bool) -> None:
        """Detect pulses above/below baseline."""
        self.threshold = self.std_dev * threshold_multiplier
        
        if negative_pulses:
            mask = (self.signal - self.baseline) < -self.threshold
            amplitudes = -(self.signal[mask] - self.baseline[mask])
        else:
            mask = (self.signal - self.baseline) > self.threshold
            amplitudes = self.signal[mask] - self.baseline[mask]
            
        self.pulse_times = self.times[mask]
        self.pulse_amplitudes = amplitudes
        
    def reduce_pulses(self, max_gap: Optional[float] = None) -> None:
        """Combine adjacent pulse points into single pulses.
        
        Args:
            max_gap: Maximum time gap (ms) between adjacent points to be considered 
                    same pulse. Defaults to 1.5 * dt if None.
        """
        if max_gap is None:
            max_gap = 1.5 * self.dt
            
        self.pulses = []
        i = 0
        while i < len(self.pulse_times):
            # Start new pulse
            pulse_points = [i]
            
            # Add adjacent points
            while (i < len(self.pulse_times) - 1 and 
                   self.pulse_times[i + 1] - self.pulse_times[i] <= max_gap):
                i += 1
                pulse_points.append(i)
                
            # Create pulse object
            times = self.pulse_times[pulse_points]
            amplitudes = self.pulse_amplitudes[pulse_points]
            
            pulse = PhotonPulse(
                time=np.mean(times),
                area=np.sum(amplitudes),
                width=times[-1] - times[0]
            )
            self.pulses.append(pulse)
            i += 1
            
    @property
    def pulse_count(self) -> int:
        """Return number of detected pulses."""
        return len(self.pulses)
    
    def get_pulse_arrays(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return arrays of pulse times and areas.
        
        Returns:
            Tuple containing (times_array, areas_array)
        """
        if not hasattr(self, 'pulses'):
            raise RuntimeError("Must call reduce_pulses() before getting arrays")
            
        times = np.array([p.time for p in self.pulses])
        areas = np.array([p.area for p in self.pulses])
        return times, areas

#===============================================================================================================================================
def calculate_stft(time_array, signal, samples_per_fft, overlap_fraction, window, freq_min=None, freq_max=None):
    # Calculate basic parameters
    dt = time_array[1] - time_array[0]  # Time step
    
    # Calculate overlap and hop size
    overlap = int(samples_per_fft * overlap_fraction)
    hop = samples_per_fft - overlap
    
    # Calculate resolutions
    time_resolution = dt * hop  # Time between successive FFTs
    freq_resolution = 1.0 / (dt * samples_per_fft)  # Frequency resolution
    
    # Create window function
    if window.lower() == 'hanning':
        win = np.hanning(samples_per_fft)
    elif window.lower() == 'blackman':
        win = np.blackman(samples_per_fft)
    else:
        win = np.ones(samples_per_fft)
    
    # Pad signal if necessary
    pad_length = (samples_per_fft - len(signal)) % hop
    if pad_length > 0:
        signal = np.pad(signal, (0, pad_length), mode='constant')
    
    # Create strided array of segments using numpy's stride tricks
    shape = (samples_per_fft, (len(signal) - samples_per_fft) // hop + 1)
    strides = (signal.strides[0], signal.strides[0] * hop)
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    # Apply window to all segments at once
    segments = segments.T * win
    
    # Compute FFT for all segments at once
    stft_matrix = np.fft.rfft(segments, axis=1)
    
    # Compute magnitude (normalized)
    stft_matrix = 2.0/samples_per_fft * np.abs(stft_matrix)
    
    # Create frequency array
    freq = np.fft.rfftfreq(samples_per_fft, dt)

    # Apply frequency mask if specified
    if freq_min is not None and freq_max is not None:
        freq_mask = (freq >= freq_min) & (freq <= freq_max)
        freq = freq[freq_mask]
        stft_matrix = stft_matrix[:, freq_mask]

    return  freq, stft_matrix, time_resolution, freq_resolution
#===============================================================================================================================================
# def get_Bdot_calibration(filepath):
#     data_dict = read_NA_data(filepath)
    
#     return data_dict

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":
    ifn = r"E:\x-ray\20241102\C3--E-ring-p30-z13-x200-xray--00011.trc"
    xray_data, tarr_x = read_trc_data(ifn)

    time_ms = tarr_x * 1000
    detector = Photons(time_ms, xray_data, threshold_multiplier=7, cutoff_freq=0.00001)
    detector.reduce_pulses()
    pulse_times, pulse_areas = detector.get_pulse_arrays()
    
    # Plot 1: Original signal and baseline
    plot_original_and_baseline(time_ms, xray_data, detector)
    plt.show()