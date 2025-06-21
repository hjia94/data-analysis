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
def first_and_last_zerocrossings(cur):
    """
    Find the first and last positive-going zero crossings in approximately sinusoidal data.
    
    Parameters:
    data (numpy array): Input sinusoidal data (e.g., current waveform).
    lpf_gsmooth_interval (int): Smoothing interval for the low-pass filter.
    
    Returns:
    int: Index of the first positive-going zero crossing.
    int: Index of the last positive-going zero crossing.
    float: Fractional number of points in one cycle.
    """

    n = np.size(cur)
    lpf_gsmooth_interval = 25
    scur = ndimage.gaussian_filter1d(cur, lpf_gsmooth_interval)
    first = 0
    last = 0
    i = lpf_gsmooth_interval * 4
    count = 0
    NPeriod = 0
    while i < n - lpf_gsmooth_interval*4:
        if scur[i+1] > 0  and  scur[i] <= 0:
            # when we find a positive-going zero-crossing
            if first == 0:
                first = i        # first one
            else:
                last = i         # last one found
                NPeriod = (last - first) / count
                i += lpf_gsmooth_interval    # try to avoid local noise
                i += int(0.7*NPeriod)        # extra advance
            count += 1
        i += 1

    return first, last, NPeriod    # note: NPeriod is not an integer
#===========================================================================================================
def find_all_zerocrossing(signal, direction='all'):

    """Simple zero crossing detector"""
    # Center signal around zero
    signal = signal - np.median(signal)  # Using median instead of mean
    
    # Find zero crossings
    zero_crossings = []
    for i in range(1, len(signal)):
        if signal[i-1] * signal[i] <= 0:  # Zero crossing between adjacent points
            # Determine crossing direction
            is_positive = signal[i-1] < 0 and signal[i] >= 0
            if direction == 'all' or (direction == 'positive' and is_positive) or (direction == 'negative' and not is_positive):
                # Simply return the index without interpolation
                zero_crossings.append(i)
                
    return np.array(zero_crossings)

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
        data: pandas Series or numpy array
        window_size: Size of the rolling window (in samples)
        quantile: Quantile for baseline estimation (e.g., 0.1 for the 10th percentile)
    Returns:
        baseline: Smoothed baseline
    """
    return data.rolling(window=window_size, center=True).quantile(quantile)

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

#===========================================================================================================
def analyze_downsample_options(data, tarr, filtered_data, min_timescale_ms=1e-3, verbose=False):
    """
    Analyze different downsample rates and their effect on filtered data.
    
    Args:
        data: Original signal array
        tarr: Time array in milliseconds
        filtered_data: Filtered signal array
        min_timescale_ms: Minimum timescale to preserve in milliseconds
    
    Returns:
        int: Recommended downsample rate
    """
    # Calculate sampling rate
    dt = tarr[1] - tarr[0]
    min_samples = min_timescale_ms / dt  # minimum samples needed
    
    # Try different downsample rates
    rates = [2, 5, 10, 20, 50]
    errors = []

    if verbose:
        print(f"\nAnalyzing downsample rates (minimum {min_samples:.1e} samples needed for {min_timescale_ms:.1e} ms features):")

    for rate in rates:
        # Skip rates that would undersample the minimum timescale
        if rate > min_samples/2:  # Nyquist criterion
            print(f"Rate {rate:2d}: Too high for {min_timescale_ms:.1e} ms features")
            continue
            
        # Downsample filtered data
        filtered_down = filtered_data[::rate]
        
        # Interpolate back to original size for comparison
        filtered_up = np.interp(np.arange(len(data)), 
                              np.arange(len(filtered_down))*rate, 
                              filtered_down)
        
        # Compare with original filtered data
        error = np.abs(filtered_data - filtered_up).mean()
        errors.append((rate, error))
        if verbose:
            print(f"Rate {rate:2d}: Mean error = {error:.2e}, Samples in min_timescale = {min_samples/rate:.1f}")
    
    if errors:
        # Find best rate that preserves features
        best_rate = min(errors, key=lambda x: x[1])[0]
        if verbose:
            print(f"\nRecommended downsample rate: {best_rate}")
        return best_rate
    else:
        if verbose:
            print("\nNo suitable downsample rates found for given timescale")
        return 1

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
        baseline (NDArray): Computed baseline of the signal
        std_dev (float): Standard deviation of the baseline-subtracted signal
        dt (float): Time step between samples
        threshold (float): Detection threshold for pulses
        pulses (list[PhotonPulse]): Detected photon pulses
    """
    
    def __init__(self,
                 tarr: NDArray[np.float64], 
                 data_array: NDArray[np.float64], 
                 min_timescale: float = 5e-5,
                 tsh_mult: list[int] = [9, 100],
                 savgol_window: int = 31,
                 savgol_order: int = 3,
                 distance_mult: float = 0.002,
                 downsample_rate: Optional[int] = None,
                 debug: bool = False):
        """Initialize photon pulse detector.
        
        Args:
            tarr (NDArray[np.float64]): Time array in seconds
            data_array (NDArray[np.float64]): Signal amplitude array
            min_timescale (float): Minimum timescale to preserve in seconds
            tsh_mult (list[int]): [lower, upper] threshold multipliers for pulse detection
            savgol_window (int): Window length for Savitzky-Golay filter
            savgol_order (int): Polynomial order for Savitzky-Golay filter
            distance_mult (float): Multiplier for baseline distance calculation
            downsample_rate (Optional[int]): Manual downsample rate. If None, will be automatically determined
            debug (bool): Whether to show debug plots
        """
        if len(tarr) != len(data_array):
            raise ValueError("Time and data arrays must have same length")
        if len(tarr) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        # Convert everything to milliseconds for consistency
        self.tarr = tarr * 1000  # Convert to ms
        self.dt = (self.tarr[1] - self.tarr[0])
        self.min_timescale = min_timescale * 1000  # Convert to ms
        self.upths_mult = tsh_mult[1]
        self.lowths_mult = tsh_mult[0]
        self.distance_mult = distance_mult
        self.debug = debug

        # Initial signal filtering
        print("Applying Savitzky-Golay filter...")
        self.filtered_data = signal.savgol_filter(data_array, window_length=savgol_window, polyorder=savgol_order)
        
        # Determine downsample rate if not provided
        if downsample_rate is None:
            print("Analyzing optimal downsample rate...")
            downsample_rate = analyze_downsample_options(data_array, self.tarr, self.filtered_data,
                                                       min_timescale_ms=self.min_timescale,
                                                       verbose=self.debug)
            print(f"Downsample rate: {downsample_rate}")
        self.downsample_rate = downsample_rate

        # Downsample filtered data
        self.tarr_ds = self.tarr[::self.downsample_rate]
        self.data_ds = self.filtered_data[::self.downsample_rate]
        self.dt = (self.tarr_ds[1] - self.tarr_ds[0])
        self.baseline_dis = int(self.min_timescale / self.dt * self.distance_mult)
        self.peak_detect_dis = int(self.min_timescale / self.dt)
        
        print("Computing baseline...")
        self._compute_baseline()
        print("Computing thresholds...")
        self._compute_thresholds()
        print("Detecting pulses...")
        self._detect_pulses()
        
    def _compute_baseline(self) -> None:
        """Compute baseline using envelope detection or constant baseline."""
        
        if self.distance_mult == 0:
            # Use constant baseline: average of first 5% and last 5% of data
            n_samples = len(self.data_ds)
            first_5_percent = self.data_ds[:int(n_samples * 0.05)]
            last_5_percent = self.data_ds[-int(n_samples * 0.05):]
            
            # Calculate constant baseline value
            baseline_value = (np.mean(first_5_percent) + np.mean(last_5_percent)) / 2
            
            # Create constant baseline array
            self.baseline = np.full_like(self.data_ds, baseline_value)
            self.baseline_subtracted = self.data_ds - self.baseline
            
            print(f"Using constant baseline: {baseline_value:.6f}")
            
        else:
            # Use envelope detection method (original implementation)
            # Get upper envelope points
            _, harr = hl_envelopes_idx(self.data_ds, dmin=1, dmax=self.baseline_dis, split=False)
            
            # Get noise amplitude from first 0.1% of data
            noise_sample = self.data_ds[:int(len(self.data_ds)*0.001)]
            noise_amplitude = (np.max(np.abs(noise_sample)) - np.min(np.abs(noise_sample))) / 2
            
            # Interpolate baseline using upper envelope points
            self.baseline = np.interp(np.arange(len(self.data_ds)), harr, self.data_ds[harr]) - noise_amplitude
            self.baseline_subtracted = self.baseline - self.data_ds

        if self.debug:
            plt.figure()
            plt.plot(self.tarr_ds, self.data_ds, label='Original')
            plt.plot(self.tarr_ds, self.baseline, label='Baseline')
            # plt.plot(self.tarr_ds, self.baseline_subtracted, label='Baseline Subtracted')
            plt.legend(loc='lower right')
            plt.show()
        
    def _compute_thresholds(self) -> None:
        """Compute detection thresholds based on noise statistics."""

        noise_sample = self.baseline_subtracted[:int(len(self.baseline_subtracted)*0.001)]
        self.noise_mean = np.mean(noise_sample)
        self.noise_std = np.std(noise_sample)
        
        self.lower_threshold = self.noise_mean + self.lowths_mult * self.noise_std
        self.upper_threshold = self.noise_mean + self.upths_mult * self.noise_std  # For detecting oversized pulses
        
    def _detect_pulses(self) -> None:
        """Detect pulses using signal_find_peaks."""
        
        # Find peaks above lower threshold
        peak_indices, _ = signal.find_peaks(self.baseline_subtracted, 
                                          height=self.lower_threshold,
                                          distance=self.peak_detect_dis)
        
        # Remove peaks that exceed upper threshold and nearby peaks
        mask = np.ones(len(peak_indices), dtype=bool)
        for i, idx in enumerate(peak_indices):
            if self.baseline_subtracted[idx] > self.upper_threshold:
                # Remove peaks within extended window around large peaks
                nearby_mask = np.abs(peak_indices - idx) <= self.peak_detect_dis*20
                mask[nearby_mask] = False
                
        # Apply mask to keep only valid peaks
        self.peak_indices = peak_indices[mask]
        self.pulse_times = self.tarr_ds[self.peak_indices]
        self.pulse_amplitudes = self.baseline_subtracted[self.peak_indices]

        if self.debug:
            plt.figure()
            plt.plot(self.tarr_ds, self.baseline_subtracted)
            plt.axhline(y=self.lower_threshold, color='g', linestyle='--', label='Lower Threshold')
            plt.axhline(y=self.upper_threshold, color='r', linestyle='--', label='Upper Threshold')
            plt.scatter(self.pulse_times, self.pulse_amplitudes, color='red')
            plt.xlabel('Time (ms)')
            plt.ylabel('Signal')
            plt.title('Detected Pulses')
            plt.legend(loc='upper left')
            plt.show()

    def reduce_pulses(self) -> None:
        """Process detected peaks into pulse objects."""
        self.pulses = []
        
        # For each detected peak
        for i, peak_idx in enumerate(self.peak_indices):
            # Find pulse boundaries (where signal crosses noise mean or changes direction)
            left_idx = peak_idx
            while (left_idx > 0 and 
                   self.baseline_subtracted[left_idx-1] > self.noise_mean):
                left_idx -= 1
                
            right_idx = peak_idx
            while (right_idx < len(self.baseline_subtracted)-1 and 
                   self.baseline_subtracted[right_idx+1] > self.noise_mean):
                right_idx += 1
            
            # Get pulse region data
            pulse_times = self.tarr_ds[left_idx:right_idx+1]
            pulse_amplitudes = self.baseline_subtracted[left_idx:right_idx+1]
            
            # Create pulse object
            pulse = PhotonPulse(
                time=self.tarr_ds[peak_idx],
                area=np.trapz(pulse_amplitudes, pulse_times),
                width=pulse_times[-1] - pulse_times[0]
            )
            self.pulses.append(pulse)

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
    
def counts_per_bin(tarr, amplitudes, bin_width=0.2, amplitude_min=None, amplitude_max=None):
    """
    Calculate number of pulses in each time bin with optional amplitude filtering.
    
    Args:
        tarr (array-like): Array of pulse times
        amplitudes (array-like): Array of pulse amplitudes
        bin_width (float): Width of time bins
        amplitude_min (float, optional): Minimum amplitude threshold for counting pulses
        amplitude_max (float, optional): Maximum amplitude threshold for counting pulses
    Returns:
        tuple: (bin_centers, counts) arrays where counts shows number of pulses in each bin
    """
    # Check if we have any pulses
    if len(tarr) == 0:
        return np.array([]), np.array([])
    
    # Apply amplitude filtering if requested
    if amplitude_min is not None or amplitude_max is not None:
        mask = np.ones(len(tarr), dtype=bool)
        
        if amplitude_min is not None:
            mask &= (amplitudes >= amplitude_min)
            
        if amplitude_max is not None:
            mask &= (amplitudes <= amplitude_max)
            
        tarr = tarr[mask]
    
    # If no pulses remain after filtering, return empty arrays
    if len(tarr) == 0:
        return np.array([]), np.array([])
    
    # Create time bins
    time_min = np.min(tarr)
    time_max = np.max(tarr)
    n_bins = max(1, int((time_max - time_min) / bin_width))
    bins = np.linspace(time_min, time_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Count pulses in each bin
    counts, _ = np.histogram(tarr, bins=bins)
    
    return bin_centers, counts
#===============================================================================================================================================
def calculate_stft(time_array, data_arr, freq_bins=100, overlap_fraction=0.1, window='hanning', freq_min=None, freq_max=None):
    """
    Calculate Short-Time Fourier Transform with specified number of frequency bins.
    
    Args:
        time_array: Array of time points
        data_arr: Signal data array
        freq_bins: Number of frequency bins desired between freq_min and freq_max
        overlap_fraction: Fraction of overlap between consecutive FFT windows
        window: Window function ('hanning', 'blackman', or None)
        freq_min: Minimum frequency to include in output (Hz)
        freq_max: Maximum frequency to include in output (Hz)
        
    Returns:
        freq: Frequency array
        stft_matrix: STFT magnitude matrix
        stft_time: Time array corresponding to each FFT segment
        freq_resolution: Frequency resolution of the STFT
    """
    # Calculate basic parameters
    dt = time_array[1] - time_array[0]  # Time step
    fs = 1.0 / dt  # Sampling frequency
    
    # Calculate samples_per_fft based on desired frequency bins
    if freq_min is not None and freq_max is not None:
        # Calculate samples needed for desired frequency resolution
        desired_freq_resolution = (freq_max - freq_min) / freq_bins
        samples_per_fft = int(fs / desired_freq_resolution)
        
        # Ensure samples_per_fft is large enough to cover the frequency range
        nyquist_freq = fs / 2
        if freq_max > nyquist_freq:
            print(f"Warning: Maximum frequency {freq_max/1e6:.1f} MHz exceeds Nyquist frequency {nyquist_freq/1e6:.1f} MHz")
            freq_max = nyquist_freq
        
        # Minimum samples needed for the requested frequency range
        min_samples_needed = int(2 * fs / (freq_max - freq_min) * freq_bins)
        
        if samples_per_fft < min_samples_needed:
            samples_per_fft = min_samples_needed
            print(f"Increased samples_per_fft to {samples_per_fft} to accommodate frequency range")
        
        # Ensure samples_per_fft is even (for FFT efficiency)
        if samples_per_fft % 2 != 0:
            samples_per_fft += 1
            
        # Ensure samples_per_fft is not too large
        max_samples = len(data_arr) // 2
        if samples_per_fft > max_samples:
            samples_per_fft = max_samples
            print(f"Warning: Reduced samples_per_fft to {samples_per_fft} due to data length constraints")
    else:
        # Default to a reasonable fraction of the data length if no freq range specified
        samples_per_fft = min(1024, len(data_arr) // 4)
    
    # Calculate overlap and hop size
    overlap = int(samples_per_fft * overlap_fraction)
    hop = samples_per_fft - overlap
    
    # Calculate time resolution
    time_resolution = dt * hop  # Time between successive FFTs
    freq_resolution = fs / samples_per_fft  # Frequency resolution
    
    # Create window function
    if window.lower() == 'hanning':
        win = np.hanning(samples_per_fft)
    elif window.lower() == 'blackman':
        win = np.blackman(samples_per_fft)
    else:
        win = np.ones(samples_per_fft)
    
    # Pad signal if necessary
    pad_length = (samples_per_fft - len(data_arr) % hop) % hop
    if pad_length > 0:
        data_arr = np.pad(data_arr, (0, pad_length), mode='constant')
        # Also pad the time array (extrapolate time values)
        if len(time_array) < len(data_arr):
            dt = time_array[1] - time_array[0]
            extra_times = np.arange(len(time_array), len(data_arr)) * dt + time_array[-1]
            time_array = np.concatenate([time_array, extra_times])
    
    # Calculate indices of the start of each segment
    num_segments = (len(data_arr) - samples_per_fft) // hop + 1
    segment_indices = np.arange(num_segments) * hop
    
    # Calculate the middle time point for each segment
    mid_indices = segment_indices + samples_per_fft // 2
    stft_time = time_array[mid_indices]
    
    # Create strided array of segments using numpy's stride tricks
    shape = (samples_per_fft, num_segments)
    strides = (data_arr.strides[0], data_arr.strides[0] * hop)
    segments = np.lib.stride_tricks.as_strided(data_arr, shape=shape, strides=strides)
    
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
        if not any(freq_mask):
            print("Warning: No frequencies in the requested range. Check your frequency limits and sampling rate.")
            # Use all available frequencies as fallback
            freq_mask = np.ones_like(freq, dtype=bool)
        freq = freq[freq_mask]
        stft_matrix = stft_matrix[:, freq_mask]
    
    print(f"STFT calculated with {samples_per_fft} samples per FFT window")
    print(f"Frequency range: {freq[0]/1e6:.1f} MHz to {freq[-1]/1e6:.1f} MHz")
    print(f"Generated {len(stft_time)} time points for {stft_matrix.shape[0]} FFT segments")

    return freq, stft_matrix, stft_time
#===============================================================================================================================================
# def get_Bdot_calibration(filepath):
#     data_dict = read_NA_data(filepath)
    
#     return data_dict

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================
