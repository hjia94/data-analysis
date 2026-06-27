'''
Generic digital-signal-processing helpers (no plasma knowledge).

Filters, zero-crossing detection, envelopes, downsample analysis, and STFT.
These operate on plain numpy arrays and are reused across diagnostics.

Author: Jia Han
Date: 2024-02-28
'''

import numpy as np
from scipy import signal, ndimage


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


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


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

    # Store original data length before padding
    original_length = len(data_arr)

    # Pad signal if necessary
    pad_length = (samples_per_fft - len(data_arr) % hop) % hop
    if pad_length > 0:
        data_arr = np.pad(data_arr, (0, pad_length), mode='constant')

    # Calculate indices of the start of each segment
    num_segments = (len(data_arr) - samples_per_fft) // hop + 1
    segment_indices = np.arange(num_segments) * hop

    # Calculate the middle time point for each segment, but only for segments within original data
    mid_indices = segment_indices + samples_per_fft // 2

    # Only keep segments where the middle point is within the original data range
    valid_segments = mid_indices < original_length
    segment_indices = segment_indices[valid_segments]
    mid_indices = mid_indices[valid_segments]
    num_segments = len(segment_indices)

    stft_time = time_array[mid_indices]

    # Create strided array of segments using numpy's stride tricks
    # Use only the valid segments that fit within original data
    shape = (num_segments, samples_per_fft)
    strides = (data_arr.strides[0] * hop, data_arr.strides[0])
    segments = np.lib.stride_tricks.as_strided(data_arr, shape=shape, strides=strides)

    # Apply window to all segments at once
    segments = segments * win

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
