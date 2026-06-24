'''
PhotonPulse and Photons classes for pulse-width analysis.

Original code from Pat, written for McPherson spectrometer data. Performs pulse
detection and analysis on photon pulses. Can be used for data obtained from any
PMT in general. The PhotonPulse and Photons classes analyze photon pulses in
time series data, typically from PMT signals.

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

import numpy as np
from scipy import signal

from dataclasses import dataclass
from typing import Tuple, Optional
from numpy.typing import NDArray

from data_analysis.signal.core import analyze_downsample_options, hl_envelopes_idx


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
            import matplotlib.pyplot as plt
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
            import matplotlib.pyplot as plt
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
