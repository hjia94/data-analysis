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
from scipy import signal, ndimage
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

def fast_gaussian_filter(data, sigma):
    # Create Gaussian kernel
    kernel_size = int(6 * sigma)  # 3 sigma on each side
    kernel = np.exp(-np.linspace(-3, 3, kernel_size)**2 / 2)
    kernel = kernel / kernel.sum()  # normalize
    
    # Use FFT-based convolution
    return signal.fftconvolve(data, kernel, mode='same')

def low_pass_filter(data, cutoff_freq):
        # Design a low-pass Butterworth filter
        b, a = signal.butter(2, cutoff_freq, btype='low')
        
        return signal.filtfilt(b, a, data)

def rolling_baseline(data, window_size=1000, quantile=0.1):
    """
    Estimate the baseline using a rolling window and a lower quantile.
    Args:
        signal: Input signal as a pandas Series or numpy array
        window_size: Size of the rolling window (in samples)
        quantile: Quantile for baseline estimation (e.g., 0.1 for the 10th percentile)
    Returns:
        baseline: Smoothed baseline of the signal as a pandas Series
    """
    return data.rolling(window=window_size, center=True).quantile(quantile)

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
        baseline (NDArray): Computed baseline of the signal
        std_dev (float): Standard deviation of the baseline-subtracted signal
        dt (float): Time step between samples
        threshold (float): Detection threshold for pulses
        pulses (list[PhotonPulse]): Detected photon pulses
    """
    
    def __init__(self, 
                 times: NDArray[np.float64], 
                 data_array: NDArray[np.float64], 
                 threshold_multiplier: float = 7.0,
                 baseline_filter_value: float = 10001,
                 pulse_filter_value: float = 11,
                 baseline_filter_type: str = 'savgol',
                 pulse_filter_type: str = 'savgol'):
        """Initialize photon pulse detector.
        
        Args:
            times: Time array in milliseconds
            data_array: Signal amplitude array
            threshold_multiplier: Number of standard deviations above baseline for detection
            baseline_filter_value: Filter parameter for baseline - sigma for gaussian, cutoff for butterworth, window for savgol
            pulse_filter_value: Filter parameter for pulse smoothing - sigma for gaussian, cutoff for butterworth, window for savgol
            baseline_filter_type: Type of filter to use for baseline ('gaussian', 'butterworth', or 'savgol')
            pulse_filter_type: Type of filter to use for pulse smoothing ('gaussian', 'butterworth', or 'savgol')
        
        Raises:
            ValueError: If input arrays have different lengths or are empty
        """
        if len(times) != len(data_array):
            raise ValueError("Time and data arrays must have same length")
        if len(times) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        self.times = times
        self.data = data_array
        
        # Ensure window lengths are odd for Savitzky-Golay filter
        if baseline_filter_type == 'savgol':
            self.baseline_filter_value = int(baseline_filter_value) // 2 * 2 + 1
        else:
            self.baseline_filter_value = baseline_filter_value
            
        if pulse_filter_type == 'savgol':
            self.pulse_filter_value = int(pulse_filter_value) // 2 * 2 + 1
        else:
            self.pulse_filter_value = pulse_filter_value
            
        self.baseline_filter_type = baseline_filter_type
        self.pulse_filter_type = pulse_filter_type
        
        # Calculate time step
        self.dt = np.mean(np.diff(self.times))

        print("Computing baseline...")
        self._compute_baseline()
        print("Computing threshold...")
        self._compute_threshold(threshold_multiplier)
        print("Detecting pulses...")
        self._detect_pulses()
        
    def _compute_baseline(self) -> None:
        """Compute baseline of the signal using the chosen filter."""
        # Apply baseline filter to get the slow-varying baseline
        if self.baseline_filter_type == 'gaussian':
            self.baseline = fast_gaussian_filter(self.data, sigma=self.baseline_filter_value)
        elif self.baseline_filter_type == 'butterworth':
            self.baseline = low_pass_filter(self.data, self.baseline_filter_value)
        elif self.baseline_filter_type == 'savgol':
            self.baseline = signal.savgol_filter(self.data, 
                                               window_length=self.baseline_filter_value, 
                                               polyorder=3)
        else:
            # Take last 5% of data points for baseline calculation
            n_points = len(self.data)
            n1 = int(0.95 * n_points)
            self.baseline = np.full_like(self.data, np.mean(self.data[n1:]))

        # Subtract baseline from signal
        self.baseline_subtracted = self.data - self.baseline
        
    def _compute_threshold(self, threshold_multiplier: float) -> None:
        """Compute detection threshold based on signal standard deviation.
        
        Args:
            threshold_multiplier: Number of standard deviations for threshold
        """
        # Compute standard deviation from last 5% of baseline-subtracted data
        n_points = len(self.baseline_subtracted)
        n1 = int(0.95 * n_points)
        self.std_dev = np.std(self.baseline_subtracted[n1:])
        self.threshold = self.std_dev * threshold_multiplier
        
    def _detect_pulses(self) -> None:
        """Detect pulses above threshold in filtered signal."""
        # Apply pulse smoothing filter to reduce high-frequency noise
        if self.pulse_filter_type == 'gaussian':
            filtered_signal = fast_gaussian_filter(self.baseline_subtracted, 
                                                sigma=self.pulse_filter_value)
        elif self.pulse_filter_type == 'butterworth':
            filtered_signal = low_pass_filter(self.baseline_subtracted, 
                                           self.pulse_filter_value)
        elif self.pulse_filter_type == 'savgol':
            filtered_signal = signal.savgol_filter(self.baseline_subtracted,
                                                window_length=self.pulse_filter_value,
                                                polyorder=3)
        else:
            filtered_signal = self.baseline_subtracted
            
        # Detect pulses above threshold in filtered signal
        mask = filtered_signal > self.threshold
        
        # Get amplitudes from baseline-subtracted signal for true pulse heights
        self.pulse_amplitudes = self.baseline_subtracted[mask]
        self.pulse_times = self.times[mask]
        
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
