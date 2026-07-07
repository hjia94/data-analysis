'''
Generic digital-signal-processing helpers (no plasma knowledge).

Filters, zero-crossing detection, envelopes, downsample analysis, and STFT.
These operate on plain numpy arrays and are reused across diagnostics.

Author: Jia Han
Date: 2024-02-28
'''

import numpy as np
from scipy import signal, ndimage

# np.trapz was renamed np.trapezoid in numpy 2.0 (np.trapz deprecated).
# Prefer the new name, fall back for numpy < 2.0.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def first_and_last_zerocrossings(cur):
    """
    Find the first and last positive-going zero crossings in approximately sinusoidal data.

    Kept for reuse (past sweep-timing analyses) even when no current experiment calls it.

    Parameters:
    cur (numpy array): Input sinusoidal data (e.g., current waveform), assumed
        centered around zero. The data is Gaussian-smoothed with a hardcoded
        25-sample interval before crossing detection to suppress noise.

    Returns:
    int: Index of the first positive-going zero crossing.
    int: Index of the last positive-going zero crossing.
    float: Fractional number of points in one cycle (0 if < 2 crossings found).
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


def find_all_zerocrossing(data, direction='all'):
    """Simple zero crossing detector.

    Centers ``data`` around its median, then returns the indices ``i`` where the
    signal crosses zero between samples ``i-1`` and ``i`` (the right-hand sample
    of each crossing pair; no sub-sample interpolation). ``direction`` selects
    ``'positive'`` (negative -> non-negative), ``'negative'``, or ``'all'``
    crossings. Returns an int index array.
    """
    s = np.asarray(data, float)
    s = s - np.median(s)  # Using median instead of mean

    # Zero crossing between adjacent points: sign product <= 0
    cross = s[:-1] * s[1:] <= 0
    if direction in ('positive', 'negative'):
        is_positive = (s[:-1] < 0) & (s[1:] >= 0)
        cross &= is_positive if direction == 'positive' else ~is_positive

    return np.nonzero(cross)[0] + 1


def fast_gaussian_filter(data, sigma):
    """Gaussian smoothing via FFT convolution (fast on long traces).

    ``sigma`` is the Gaussian width in samples; the kernel spans +-3 sigma
    (odd length, minimum 3, so tiny sigmas still yield a valid kernel and the
    output stays aligned with the input). FFT convolution
    (``scipy.signal.fftconvolve``) is much faster than direct convolution for
    the multi-million-sample traces this is meant for. Returns the smoothed
    array (same length, ``mode='same'``).
    """
    # Create Gaussian kernel: odd size >= 3 (even kernels shift the output by
    # half a sample; size 0 would divide by zero in the normalization)
    kernel_size = max(int(6 * sigma) | 1, 3)  # ~3 sigma on each side
    kernel = np.exp(-np.linspace(-3, 3, kernel_size)**2 / 2)
    kernel = kernel / kernel.sum()  # normalize

    # Use FFT-based convolution
    return signal.fftconvolve(data, kernel, mode='same')


def low_pass_filter(data, cutoff_freq, fs=None):
    """Zero-phase 2nd-order Butterworth low-pass filter.

    With ``fs`` given (sampling frequency), ``cutoff_freq`` is in the same
    units (e.g. both in Hz). With ``fs=None`` (legacy behavior),
    ``cutoff_freq`` must already be normalized to the Nyquist frequency
    (0..1). Filters forward and backward (``filtfilt``) so the output has no
    phase shift. Returns the filtered array.
    """
    b, a = signal.butter(2, cutoff_freq, btype='low', fs=fs)
    return signal.filtfilt(b, a, data)


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter in SOS form.

    ``lowcut``/``highcut`` are the band edges and ``fs`` the sampling
    frequency, all in the same units (e.g. Hz). Returns second-order sections
    (numerically stable at higher orders, unlike the ``(b, a)`` form) --
    apply with ``scipy.signal.sosfiltfilt(sos, data)`` for zero-phase
    filtering.
    """
    return signal.butter(order, [lowcut, highcut], btype='band',
                         output='sos', fs=fs)


def rolling_baseline(data, window_size=1000, quantile=0.1):
    """
    Estimate the baseline using a centered rolling window and a lower quantile.
    Args:
        data: numpy array (or anything np.asarray accepts, e.g. pandas Series)
        window_size: Size of the rolling window (in samples)
        quantile: Quantile for baseline estimation (e.g., 0.1 for the 10th percentile)
    Returns:
        baseline: numpy array, same length as data (edges use the nearest
            in-bounds samples rather than NaN)
    """
    return ndimage.percentile_filter(np.asarray(data, float),
                                     percentile=quantile * 100,
                                     size=window_size, mode='nearest')


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


def downsample_stride(t, x, q):
    """Keep every ``q``-th sample (no filtering; aliases).

    Cheapest downsample.  Fluctuations above the new Nyquist fold back into the
    band, so it's fine for a quick look at the slow envelope but misleading for
    spectral content.  Returns ``(t_ds, x_ds)``.
    """
    return t[::q], x[::q]


def downsample_blockmean(t, x, q):
    """Mean of each non-overlapping block of ``q`` samples (boxcar LPF + decimate).

    A crude boxcar low-pass then decimate: anti-aliases somewhat and is robust,
    so it tracks the mean level / slow structure well, but the boxcar has poor
    stopband rejection.  Returns ``(t_ds, x_ds)``.
    """
    n = (x.size // q) * q
    xb = x[:n].reshape(-1, q).mean(axis=1)
    tb = t[:n:q][: xb.size]   # uniform time axis -- stride lines up with the blocks
    return tb, xb


def downsample_decimate(t, x, q):
    """scipy polyphase FIR-filtered decimation (anti-aliased).

    Proper anti-aliasing filter then downsample -- the principled choice when the
    downsampled trace will be FFT'd; preserves in-band fluctuations without
    aliasing.  ``scipy.signal.decimate`` caps the factor per call (FIR order
    grows with ``q``), so a large ``q`` is split into a chain of smaller steps.
    Returns ``(t_ds, x_ds)``.
    """
    xq = x
    for step in _decimate_factor_chain(q):
        xq = signal.decimate(xq, step, ftype="fir", zero_phase=True)
    return t[::q][: xq.size], xq


def _decimate_factor_chain(q, cap=12):
    """Split a large decimation factor into steps each <= ``cap`` (decimate FIR limit)."""
    steps = []
    while q > cap:
        f = next((d for d in range(cap, 1, -1) if q % d == 0), cap)
        steps.append(f)
        q //= f
    if q > 1:
        steps.append(q)
    return steps


# Generic downsamplers keyed by display name, so callers can iterate over all
# methods (e.g. to overlay them for comparison).  Colors / labels for plotting
# are a display choice and belong at the call site, not here.
DOWNSAMPLE_METHODS = {
    "stride": downsample_stride,
    "block mean": downsample_blockmean,
    "decimate(FIR)": downsample_decimate,
}


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
            if verbose:
                print(f"Rate {rate:2d}: Too high for {min_timescale_ms:.1e} ms features")
            continue

        # Downsample filtered data
        _, filtered_down = downsample_stride(tarr, filtered_data, rate)

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


def calculate_stft(time_array, data_arr, freq_bins=100, overlap_fraction=0.1, window='hanning', freq_min=None, freq_max=None, verbose=False):
    """
    Calculate Short-Time Fourier Transform with specified number of frequency bins.

    Args:
        time_array: Array of time points
        data_arr: Signal data array
        freq_bins: Number of frequency bins desired between freq_min and freq_max
        overlap_fraction: Fraction of overlap between consecutive FFT windows
        window: Window function ('hanning', 'blackman', or None for rectangular)
        freq_min: Minimum frequency to include in output (Hz)
        freq_max: Maximum frequency to include in output (Hz)
        verbose: Print the resolved FFT parameters (warnings always print)

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
            if verbose:
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

    # Create window function (None or unrecognized -> rectangular)
    wname = (window or '').lower()
    if wname == 'hanning':
        win = np.hanning(samples_per_fft)
    elif wname == 'blackman':
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

    if verbose:
        print(f"STFT calculated with {samples_per_fft} samples per FFT window")
        print(f"Frequency range: {freq[0]/1e6:.1f} MHz to {freq[-1]/1e6:.1f} MHz")
        print(f"Generated {len(stft_time)} time points for {stft_matrix.shape[0]} FFT segments")

    return freq, stft_matrix, stft_time


def clip_time_window(tarr, tmin_ms, tmax_ms):
    """Index range ``[i0, i1)`` of ``tarr`` (seconds) inside ``[tmin, tmax]`` ms.

    The standard "trim the shot stack to the analysis window" step: slice with
    ``stack[:, i0:i1]`` before handing the stack to a spectrum helper. Raises
    ``ValueError`` if the window selects fewer than 2 samples.
    """
    i0, i1 = np.searchsorted(tarr, [tmin_ms * 1e-3, tmax_ms * 1e-3])
    if i1 - i0 < 2:
        raise ValueError(
            f"window {tmin_ms}-{tmax_ms} ms selects < 2 samples (i0={i0}, i1={i1})")
    return int(i0), int(i1)


def finite_row_mask(*stacks):
    """Boolean mask of rows (shots) that are all-finite in every given stack.

    The row filter every shot-ensemble helper applies before averaging: NaN rows
    mark unreadable shots and must not poison the ensemble.
    """
    good = np.all(np.isfinite(stacks[0]), axis=1)
    for s in stacks[1:]:
        good &= np.all(np.isfinite(s), axis=1)
    return good


def line_average(profile, x):
    """Chord (line) average of ``profile(x)``: ``trapezoid(profile, x) / span``.

    NaN-tolerant: integrates over the finite subset of points (a probe scan
    routinely has failed-fit NaNs); fewer than two finite points -> ``nan``.
    """
    profile = np.asarray(profile, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(profile) & np.isfinite(x)
    if ok.sum() < 2:
        return np.nan
    return trapezoid(profile[ok], x[ok]) / (x[ok][-1] - x[ok][0])


def amplitude_spectrum(x, dt, detrend=True, drop_dc=True):
    """Single-sided amplitude spectrum of a 1-D trace.

    ``x`` is a 1-D signal sampled every ``dt`` seconds. Returns ``(freq, amp)``
    with ``freq`` in Hz and ``amp`` the single-sided amplitude (``2/n`` scaled).
    ``detrend`` subtracts the mean first so the DC level doesn't swamp the
    fluctuation spectrum; ``drop_dc`` drops the ``freq == 0`` bin (meaningless
    after detrend, distracting on a log plot).
    """
    x = np.asarray(x, float)
    if detrend:
        x = x - np.mean(x)
    n = x.size
    freq = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(x)) * (2.0 / n)
    if drop_dc:
        return freq[1:], amp[1:]
    return freq, amp


def avg_amplitude_spectrum(stack, dt, detrend=True, drop_dc=True):
    """Incoherent average of per-row amplitude spectra.

    ``stack`` is an ``(nrow, nsamples)`` array of traces sampled every ``dt``
    seconds. Each row is FFT'd and the single-sided amplitude spectra are
    averaged across rows. The average is *incoherent* (amplitudes, not complex):
    random row-to-row phase does not cancel, so broadband fluctuation power
    survives -- the right average for shot ensembles.

    Trim ``stack`` to the time window of interest *before* calling -- this
    function FFTs whatever it is given. ``dt`` comes from the trace's time axis
    (e.g. ``tarr[1] - tarr[0]``).

    Rows with any non-finite sample are skipped. Returns
    ``(freq, amp_mean, n_used)``. Raises ``ValueError`` if fewer than 2 samples
    are given or no finite rows remain.
    """
    stack = np.asarray(stack, float)
    dt = float(dt)

    if stack.shape[1] < 2:
        raise ValueError(f"need >= 2 samples, got {stack.shape[1]}")

    win = stack[finite_row_mask(stack)]
    if win.shape[0] == 0:
        raise ValueError("no finite rows in stack")

    if detrend:
        win = win - win.mean(axis=1, keepdims=True)
    n = win.shape[1]
    freq = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(win, axis=1)) * (2.0 / n)
    amp_mean = amp.mean(axis=0)
    if drop_dc:
        return freq[1:], amp_mean[1:], int(win.shape[0])
    return freq, amp_mean, int(win.shape[0])


# --------------------------------------------------------------------------- #
# Two-signal (cross) analysis: time-lag cross-correlation, coherence, and
# cross-phase.  Like the spectrum helpers above these are generic DSP on plain
# numpy arrays (no plasma/diagnostic knowledge); the frequency-domain pair
# (coherence + cross-phase) is the standard "how do these two probe signals
# relate, band by band" diagnostic.
# --------------------------------------------------------------------------- #

def cross_correlation(x, y, dt, normalize=True, max_lag=None):
    """Time-lag cross-correlation of two 1-D traces -> ``(lags, xcorr)``.

    ``x`` and ``y`` are 1-D signals sampled every ``dt`` seconds. Each is
    mean-subtracted (detrended) first, then correlated with an FFT
    (``scipy.signal.correlate(..., method="fft")``). ``lags`` is the lag axis in
    **seconds** (positive lag = ``y`` delayed relative to ``x``), and the lag of
    the peak is the time delay between the channels.

    ``normalize=True`` divides by ``sqrt(sum x**2 * sum y**2)`` (Pearson-style),
    so the result is in ``[-1, 1]`` and comparable across traces. ``max_lag``
    (seconds), if given, clips the returned window to ``|lag| <= max_lag``.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be matching 1-D arrays; got {x.shape}, {y.shape}")

    x = x - x.mean()
    y = y - y.mean()

    xcorr = signal.correlate(y, x, mode="full", method="fft")
    lags = signal.correlation_lags(y.size, x.size, mode="full") * dt

    if normalize:
        norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
        if norm > 0:
            xcorr = xcorr / norm

    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        lags, xcorr = lags[keep], xcorr[keep]

    return lags, xcorr


def _welch_nperseg(nsamples, nperseg):
    """Clamp ``nperseg`` to the trace length (scipy warns and clamps otherwise)."""
    return int(min(nperseg, nsamples))


def coherence_spectrum(x, y, dt, nperseg=4096, noverlap=None, detrend="constant"):
    """Magnitude-squared coherence of one 1-D pair -> ``(freq, gamma2)``.

    Thin wrapper over ``scipy.signal.coherence`` with ``fs = 1/dt``. ``gamma2(f)``
    runs 0..1: near 1 where the two channels share a frequency strongly, near 0
    where they are unrelated. Welch already averages the segments *within* this
    one trace (that is what makes a single-pair coherence meaningful); the
    across-shot ensemble estimate is :func:`avg_cross_spectrum`. ``nperseg`` is
    the segment length (trades frequency resolution vs. variance) and is clamped
    to the trace length. Returns frequency in Hz.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nperseg = _welch_nperseg(x.size, nperseg)
    freq, gamma2 = signal.coherence(x, y, fs=1.0 / dt, nperseg=nperseg,
                                    noverlap=noverlap, detrend=detrend)
    return freq, gamma2


def cross_phase_spectrum(x, y, dt, nperseg=4096, noverlap=None, detrend="constant"):
    """Cross-phase of one 1-D pair -> ``(freq, phase_rad)``.

    Wraps ``scipy.signal.csd`` (Welch cross-spectral density) and returns the
    phase ``angle(Pxy)`` in radians at each frequency. Sign convention (verified):
    ``phase`` is the phase *of* ``y`` relative to ``x`` -- if ``y`` is ``x``
    delayed (lagging) by an angle at some frequency, the phase there is negative.
    Pairs with :func:`coherence_spectrum` -- the phase is only trustworthy where
    the coherence is high. Returns frequency in Hz.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nperseg = _welch_nperseg(x.size, nperseg)
    freq, pxy = signal.csd(x, y, fs=1.0 / dt, nperseg=nperseg,
                           noverlap=noverlap, detrend=detrend)
    return freq, np.angle(pxy)


def _avg_welch_spectra(stack_x, stack_y, dt, nperseg=4096, noverlap=None,
                       detrend="constant"):
    """Shot-averaged Welch cross/auto spectra of a stack pair.

    The shared core of :func:`avg_cross_spectrum` and :func:`band_cross_spectrum`:
    validates the two ``(nshot, nsamples)`` stacks, drops rows with any non-finite
    sample, forms the per-shot Welch cross-spectrum ``Pxy`` and auto-spectra
    ``Pxx``/``Pyy`` (``scipy.signal.csd``/``welch``), and averages each **across
    shots** (the complex ``Pxy`` averaged as a complex number, so incoherent
    shot-to-shot phase cancels). Returns ``(freq, pxy_mean, pxx_mean, pyy_mean,
    n_used)`` -- the three shot-averaged spectra over the frequency axis, plus the
    number of finite shots used. Callers form ``gamma2``/``phase`` from these.
    """
    stack_x = np.asarray(stack_x, float)
    stack_y = np.asarray(stack_y, float)
    if stack_x.shape != stack_y.shape or stack_x.ndim != 2:
        raise ValueError(
            f"stacks must be matching (nshot, nsamples) arrays; got "
            f"{stack_x.shape}, {stack_y.shape}")

    good = finite_row_mask(stack_x, stack_y)
    xs, ys = stack_x[good], stack_y[good]
    if xs.shape[0] == 0:
        raise ValueError("no finite shot pairs in the stacks")

    nperseg = _welch_nperseg(xs.shape[1], nperseg)
    csd_kw = dict(fs=1.0 / dt, nperseg=nperseg, noverlap=noverlap,
                  detrend=detrend, axis=-1)

    # scipy's csd/welch work over the whole (nshot, nsamples) stack at once via
    # axis=-1, giving per-shot spectra (nshot, nf); average over shots.
    freq, pxy = signal.csd(xs, ys, **csd_kw)
    _, pxx = signal.welch(xs, **csd_kw)
    _, pyy = signal.welch(ys, **csd_kw)
    return (freq, pxy.mean(axis=0), pxx.mean(axis=0), pyy.mean(axis=0),
            int(xs.shape[0]))


def avg_cross_spectrum(stack_x, stack_y, dt, nperseg=4096, noverlap=None,
                       detrend="constant"):
    """Ensemble-averaged coherence + cross-phase over a shot stack.

    The two-signal analogue of :func:`avg_amplitude_spectrum`. ``stack_x`` and
    ``stack_y`` are ``(nshot, nsamples)`` arrays of the two channels on the
    **same** time grid (one row per shot). For each shot the Welch cross-spectrum
    ``Pxy`` and auto-spectra ``Pxx``/``Pyy`` are formed (via
    ``scipy.signal.csd``), then averaged across shots *before* taking the ratio:

        ``gamma2 = |<Pxy>|**2 / (<Pxx> * <Pyy>)``   (magnitude-squared coherence)
        ``phase  = angle(<Pxy>)``                    (cross-phase, radians)

    Averaging the complex spectra before the ratio is what makes the coherence
    non-trivial (a single segment gives ``gamma2 == 1`` identically); random
    shot-to-shot phase cancels in ``<Pxy>`` where the channels are incoherent, so
    ``gamma2`` drops there. Rows with any non-finite sample are skipped.

    Returns ``(freq, gamma2, phase, n_used)`` with ``freq`` in Hz. Raises
    ``ValueError`` if the stacks mismatch or no finite shot pair remains.
    """
    freq, pxy_mean, pxx_mean, pyy_mean, n_used = _avg_welch_spectra(
        stack_x, stack_y, dt, nperseg=nperseg, noverlap=noverlap, detrend=detrend)

    denom = pxx_mean * pyy_mean
    gamma2 = np.where(denom > 0, np.abs(pxy_mean) ** 2 / denom, 0.0)
    phase = np.angle(pxy_mean)
    return freq, gamma2, phase, n_used


def band_cross_spectrum(stack_x, stack_y, dt, fband, nperseg=4096,
                        noverlap=None, detrend="constant"):
    """Narrow-band scalar coherence + cross-phase over a shot stack.

    The single-frequency-band analogue of :func:`avg_cross_spectrum`: instead of
    returning ``gamma2``/``phase`` over the whole frequency axis, it collapses a
    band ``fband = (f_lo, f_hi)`` (Hz) to one scalar coherence and one scalar
    cross-phase per stack pair. The shot-averaged complex spectra ``<Pxy>``,
    ``<Pxx>``, ``<Pyy>`` (from :func:`_avg_welch_spectra`) are averaged over the
    band's frequency bins **before** the coherence ratio and the phase angle::

        gamma2 = |mean(<Pxy>[band])|**2 / (mean(<Pxx>[band]) * mean(<Pyy>[band]))
        phase  = angle(mean(<Pxy>[band]))

    Collapsing the complex spectra before the ratio/angle is the statistically
    correct band estimate -- averaging the per-bin ``gamma2`` values (or the
    per-bin phase) instead would mishandle the coherence bias and the +-pi phase
    wrap. Returns ``(f_center, gamma2, phase, n_used)`` -- the band-center
    frequency (Hz), the scalar coherence (0..1), the scalar cross-phase (radians),
    and the number of finite shots used. Raises ``ValueError`` if ``fband`` selects
    no frequency bin.
    """
    f_lo, f_hi = fband
    freq, pxy_mean, pxx_mean, pyy_mean, n_used = _avg_welch_spectra(
        stack_x, stack_y, dt, nperseg=nperseg, noverlap=noverlap, detrend=detrend)

    band = (freq >= f_lo) & (freq <= f_hi)
    if not band.any():
        raise ValueError(
            f"fband {f_lo}-{f_hi} Hz selects no frequency bin "
            f"(df={freq[1] - freq[0]:.3g} Hz, fmax={freq[-1]:.3g} Hz)")

    gamma2, phase = _collapse_band(pxy_mean, pxx_mean, pyy_mean, band)
    return float(freq[band].mean()), gamma2, phase, n_used


def _collapse_band(pxy_mean, pxx_mean, pyy_mean, band):
    """Collapse shot-averaged spectra over a boolean ``band`` mask to scalars.

    The statistically correct band estimate shared by :func:`band_cross_spectrum`
    and :func:`peak_cross_spectrum`: averages the complex ``<Pxy>``/``<Pxx>``/
    ``<Pyy>`` over the band's bins **before** the coherence ratio and phase angle.
    Returns ``(gamma2, phase)`` -- scalar coherence (0..1) and scalar cross-phase
    (radians). The band-center frequency is left to the caller (which already
    knows it: the fixed band center, or the located peak).
    """
    pxy_b = pxy_mean[band].mean()
    pxx_b = pxx_mean[band].mean()
    pyy_b = pyy_mean[band].mean()
    denom = pxx_b * pyy_b
    gamma2 = float(np.abs(pxy_b) ** 2 / denom) if denom > 0 else 0.0
    phase = float(np.angle(pxy_b))
    return gamma2, phase


def peak_cross_spectrum(stack_x, stack_y, dt, search_band, delta_f,
                        nperseg=4096, noverlap=None, detrend="constant"):
    """Peak-tracking scalar coherence + cross-phase over a shot stack.

    For data whose dominant mode sits at a *different* frequency at each
    measurement (e.g. a drift wave that shifts across a probe plane), a fixed band
    straddles the peak at some positions and misses it at others.  This finds the
    coherence peak within ``search_band = (f_lo, f_hi)`` (Hz) -- the frequency
    where the shared mode is strongest -- then band-averages a window of half-width
    ``delta_f`` (Hz) *centered on that peak*, so the estimate follows the mode.

    The collapse itself (:func:`_collapse_band`) is identical to
    :func:`band_cross_spectrum` -- the complex spectra are averaged before the
    ratio/angle.  ``f_peak`` is located on the shot-averaged coherence
    ``|<Pxy>|**2 / (<Pxx><Pyy>)`` (the shared mode), not on either auto-power
    alone.  Returns ``(f_peak, gamma2, phase, n_used)`` -- the located peak
    frequency (Hz, the band center actually used), the scalar coherence (0..1), the
    scalar cross-phase (radians), and the number of finite shots used.  Raises
    ``ValueError`` if ``search_band`` selects no frequency bin.
    """
    f_lo, f_hi = search_band
    freq, pxy_mean, pxx_mean, pyy_mean, n_used = _avg_welch_spectra(
        stack_x, stack_y, dt, nperseg=nperseg, noverlap=noverlap, detrend=detrend)

    search = (freq >= f_lo) & (freq <= f_hi)
    if not search.any():
        raise ValueError(
            f"search_band {f_lo}-{f_hi} Hz selects no frequency bin "
            f"(df={freq[1] - freq[0]:.3g} Hz, fmax={freq[-1]:.3g} Hz)")

    # Locate the coherence peak inside the search window (the shared mode), then
    # integrate +-delta_f around it. Coherence is the ratio of the shot-averaged
    # spectra, so incoherent power in the window does not pull the peak.
    denom = pxx_mean * pyy_mean
    coh = np.where(denom > 0, np.abs(pxy_mean) ** 2 / denom, 0.0)
    coh_search = np.where(search, coh, -np.inf)
    f_peak = freq[int(np.argmax(coh_search))]

    band = (freq >= f_peak - delta_f) & (freq <= f_peak + delta_f)
    gamma2, phase = _collapse_band(pxy_mean, pxx_mean, pyy_mean, band)
    return float(f_peak), gamma2, phase, n_used
