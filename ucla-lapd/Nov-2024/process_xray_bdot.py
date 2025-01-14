#!/usr/bin/env python3
"""
Script to process X-ray and Bdot data, creating combined plots for each shot.
Each figure will contain 4 subplots:
1. Raw X-ray signal with detected pulses
2. Photon counts histogram
3. Bdot STFT spectrogram
4. Combined STFT and counts plot
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from screeninfo import get_monitors
import tkinter as tk

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
# Add paths for custom modules
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")

from read_scope_data import read_trc_data
from data_analysis_utils import Photons, calculate_stft
from plot_utils import select_monitor, plot_stft_wt_photon_counts, plot_original_and_baseline, plot_subtracted_signal

#===========================================================================================================
#===========================================================================================================

def counts_per_bin(pulse_times, pulse_areas, bin_width_ms=0.2, amplitude_min=None, amplitude_max=None):
    """
    Calculate number of pulses in each time bin with optional amplitude filtering.
    
    Args:
        pulse_times (np.ndarray): Array of pulse arrival times in milliseconds
        pulse_areas (np.ndarray): Array of pulse areas/amplitudes
        bin_width_ms (float): Width of time bins in milliseconds
        amplitude_min (float, optional): Minimum amplitude threshold for counting pulses
        amplitude_max (float, optional): Maximum amplitude threshold for counting pulses
    
    Returns:
        tuple: (bin_centers, counts) arrays where counts shows number of pulses in each bin
    """
    # Apply amplitude thresholds if specified
    if amplitude_min is not None or amplitude_max is not None:
        # Initialize mask as all True
        mask = np.ones_like(pulse_times, dtype=bool)
        
        # Apply min threshold if specified
        if amplitude_min is not None:
            mask &= (pulse_areas >= amplitude_min)
            
        # Apply max threshold if specified
        if amplitude_max is not None:
            mask &= (pulse_areas <= amplitude_max)
            
        pulse_times = pulse_times[mask]
    
    # Create time bins
    time_min = min(pulse_times)
    time_max = max(pulse_times)
    n_bins = int((time_max - time_min) / bin_width_ms) + 1
    bins = np.linspace(time_min, time_max, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Count pulses in each bin
    counts, _ = np.histogram(pulse_times, bins=bins)
    
    return bin_centers, counts

def process_shot(date, file_number, position, monitor_idx=1):
    """Process a single shot and create a combined figure."""
    # Define file patterns
    xray_pattern = f"C{{channel}}--E-ring-{position}-xray--{file_number}.trc"
    bdot_pattern = f"C{{channel}}--E-ring-{position}-Bdot--{file_number}.trc"
    
    # Create figure with subplots
    fig = plt.figure(int(file_number), figsize=(10, 15))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.3)
    ax1, ax2, ax3 = [fig.add_subplot(g) for g in gs]
    
    # Position window
    _, x_pos, y_pos, window_width, window_height = select_monitor(monitor_idx=monitor_idx, window_scale=(0.2, 0.2))

    # Read X-ray data
    base_dir = os.path.join("E:", "x-ray", date)
    xray_data = None
    tarr_x = None
    for channel in ["2", "3"]:
        filename = xray_pattern.format(channel=channel)
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath) and channel == "3":
            xray_data, tarr_x = read_trc_data(filepath)
            xray_data = -xray_data  # Invert signal for positive pulses

    if xray_data is None or tarr_x is None:
        raise FileNotFoundError("Required X-ray data files not found")
        
    # Process X-ray data
    time_ms = tarr_x * 1000  # Convert to milliseconds
    detector = Photons(time_ms, 
                      xray_data,
                      threshold_multiplier=5,
                      baseline_filter_value=10001,
                      pulse_filter_value=11,
                      baseline_filter_type='savgol',
                      pulse_filter_type='savgol')
    detector.reduce_pulses()
    
    # Plot 1: Original signal and baseline
    plot_original_and_baseline(time_ms, xray_data, detector, ax1)
    ax1.set_title('Original Signal and Baseline')
    
    # Plot 2: Baseline-subtracted signal with pulses
    plot_subtracted_signal(time_ms, xray_data, None, detector, ax2)  # pulse_times not needed
    ax2.set_title('Baseline-subtracted Signal with Detected Pulses')
    
    # Get pulse data for counts
    pulse_times, pulse_areas = detector.get_pulse_arrays()
    
    # Calculate photon counts per bin
    bin_width_ms = 0.2
    bin_centers, counts = counts_per_bin(pulse_times, pulse_areas, bin_width_ms)
    
    # Read Bdot data
    base_dir = os.path.join("E:", "Bdot", date)
    By_P21 = None
    tarr_B = None
    for channel in ["1", "2", "3"]:
        filename = bdot_pattern.format(channel=channel)
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath) and channel == "1":
            By_P21, tarr_B = read_trc_data(filepath)
    
    if By_P21 is not None and tarr_B is not None:
        # Calculate STFT
        freq_arr, fft_arr, _, _ = calculate_stft(
            tarr_B, By_P21, 
            samples_per_fft=500000, 
            overlap_fraction=0.01, 
            window='hanning', 
            freq_min=150e6, 
            freq_max=1000e6
        )
        
        # Plot 3: STFT with photon counts
        plot_stft_wt_photon_counts(tarr_B, fft_arr, freq_arr, bin_centers, counts, fig=fig, ax=ax3)    
    
    plt.pause(0.1)
    return fig

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":
    # Configuration
    date = "20241102"
    position = "p30-z13-x200"
    file_numbers = [f"{i:05d}" for i in range(11, 12)]  # Example: 00011 to 00014
    
    print(f"Starting processing for {len(file_numbers)} shots")
    print(f"Date: {date}")
    print(f"Position: {position}")
    
    
    for file_number in file_numbers:
        try:
            fig = process_shot(date, file_number, position)

        except KeyboardInterrupt:
            print("Process terminated by KeyboardInterrupt")
            break

    print("\nScript execution completed")    
    plt.show()

