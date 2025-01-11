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
    print(f"\nProcessing shot {file_number} at position {position}")
    
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
    for channel in ["2", "3"]:
        filename = xray_pattern.format(channel=channel)
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            # if channel == "2":
            #     dipole_data, tarr_x = read_trc_data(filepath)
            if channel == "3":
                print(f"Reading x-ray data from {filename}")
                xray_data, tarr_x = read_trc_data(filepath)
                filtered_xray_data = - gaussian_filter1d(xray_data, sigma=10)

    if xray_data is None or tarr_x is None:
        raise FileNotFoundError("Required X-ray data files not found")
        
    # Process X-ray data
    time_ms = tarr_x * 1000
    detector = Photons(time_ms,
                    filtered_xray_data,
                    threshold_multiplier=2,
                    filter_type='butterworth',
                    filter_value=0.000005)
    detector.reduce_pulses()
    pulse_times, pulse_areas = detector.get_pulse_arrays()
    
    # Plot 1: Original signal and baseline
    plot_original_and_baseline(time_ms, filtered_xray_data, detector, ax1)
    
    # Plot 2: Baseline-subtracted signal with pulses
    plot_subtracted_signal(time_ms, filtered_xray_data, pulse_times, detector, ax2)
    
    # Calculate photon counts per bin
    bin_width_ms = 0.2
    bin_centers, counts = counts_per_bin(pulse_times, pulse_areas, bin_width_ms)

    total_time = max(pulse_times) - min(pulse_times)
    count_rate = len(pulse_times) / (total_time)

    # Print some statistics
    print(f"\nDetected {detector.pulse_count} pulses")
    print(f"Average pulse area: {np.mean(pulse_areas):.2f}")
    print(f"Detection threshold: {detector.threshold:.2f}")
    print(f'Average Count Rate: {count_rate:.1f} counts/ms')
    print(f'Total Counts: {len(pulse_times)}')
    print(f'Min Signal: {min(pulse_areas):.3f}')
    print(f'Max Signal: {max(pulse_areas):.3f}')


    # Clean up variables
    dipole_data = None
    xray_data = None
    detector = None
    pulse_times = None
    pulse_areas = None
    tarr_x = None

    # Read Bdot data
    base_dir = os.path.join("E:", "Bdot", date)
    
    for channel in ["1", "2", "3"]:
        filename = bdot_pattern.format(channel=channel)
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            if channel == "1":
                print(f"Reading bdot data from {filename}")
                By_P21, tarr_B = read_trc_data(filepath)
        else:
            print(f"Warning: Could not find {filepath}")

        
    # Calculate STFT
    freq_arr, fft_arr, time_resolution, freq_resolution = calculate_stft(tarr_B, By_P21, samples_per_fft=500000, overlap_fraction=0.01, window='hanning', freq_min=150e6, freq_max=1000e6)
    
    # Plot 3: Combined STFT and counts (bottom)
    plot_stft_wt_photon_counts(tarr_B, fft_arr, freq_arr, bin_centers, counts, fig=fig, ax=ax3)
    print(f'Time Res: {time_resolution*1e3:.2f} ms, Freq Res: {freq_resolution/1e6:.2f} MHz')
    
    print(f"Completed processing shot {file_number}")

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

