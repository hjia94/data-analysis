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

# Add paths for custom modules
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")

from read_scope_data import read_trc_data
from data_analysis_utils import Photons, calculate_stft
from plot_utils import plot_counts_per_bin, plot_photon_detection, select_monitor


def process_shot(date, file_number, position, monitor_idx=1):
    """Process a single shot and create a combined figure."""
    print(f"\nProcessing shot {file_number} at position {position}")
    
    # Define file patterns
    xray_pattern = f"C{{channel}}--E-ring-{position}-xray--{file_number}.trc"
    bdot_pattern = f"C{{channel}}--E-ring-{position}-Bdot--{file_number}.trc"
    
    # Use select_monitor with 20% window size
    _, x_pos, y_pos, window_width, window_height = select_monitor(
        monitor_idx=monitor_idx,
        window_scale=(0.2, 0.2)
    )

    # Create figure with subplots
    shot_num = int(file_number) 
    fig = plt.figure(shot_num, figsize=(10, 15))  # Use full file_number in title
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Set window position
    mngr = plt.get_current_fig_manager()
    try:
        # For Qt backend
        mngr.window.setGeometry(x_pos, y_pos, window_width, window_height)
    except:
        try:
            # For TkAgg backend
            mngr.window.wm_geometry(f"+{x_pos}+{y_pos}")
        except:
            try:
                # For WX backend
                mngr.window.SetPosition((x_pos, y_pos))
            except:
                print("Could not position window - unsupported backend")
    
    # Read X-ray data
    base_dir = os.path.join("E:", "x-ray", date)
    
    for channel in ["2", "3"]:
        filename = xray_pattern.format(channel=channel)
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            print(f"Reading x-ray data from {filename}")
            if channel == "2":
                dipole_data, tarr_x = read_trc_data(filepath)
            elif channel == "3":
                xray_data, tarr_x = read_trc_data(filepath)
    
    if xray_data is None or tarr_x is None:
        raise FileNotFoundError("Required X-ray data files not found")
        
    # Process X-ray data
    print("Processing X-ray pulses...")
    time_ms = tarr_x * 1000
    detector = Photons(time_ms, -xray_data, threshold_multiplier=5, negative_pulses=False)
    detector.reduce_pulses()
    pulse_times, pulse_areas = detector.get_pulse_arrays()
    
    # Plot 1: Photon detection plot (top)
    plot_photon_detection(time_ms, -xray_data, pulse_times, detector, ax=ax1)
    # Print some statistics
    print(f"\nDetected {detector.pulse_count} pulses")
    print(f"Average pulse area: {np.mean(pulse_areas):.2f}")
    print(f"Signal baseline: {detector.offset:.2f}")
    print(f"Detection threshold: {detector.threshold:.2f}")

    # Plot 2: Counts histogram (middle)
    bin_centers, counts = plot_counts_per_bin(pulse_times, pulse_areas, bin_width_ms=0.2, ax=ax2)
    
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
            print(f"Reading bdot data from {filename}")
            if channel == "1":
                By_P21, tarr_B = read_trc_data(filepath)
        else:
            print(f"Warning: Could not find {filepath}")

        
    # Calculate STFT
    freq_arr, fft_arr, time_resolution, freq_resolution = calculate_stft(tarr_B, By_P21, samples_per_fft=500000, overlap_fraction=0.01, window='hanning', freq_min=150e6, freq_max=1000e6)
    
    # Plot 3: Combined STFT and counts (bottom)
    
    # STFT plot
    im = ax3.imshow(fft_arr.T, 
                    aspect='auto',
                    origin='lower',
                    extent=[tarr_B[0]*1e3, tarr_B[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
                    interpolation='None',
                    cmap='jet')
    
    # Adjust subplot position to accommodate colorbar
    pos = ax3.get_position()
    ax3.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    
    # Add colorbar with adjusted position
    cax = fig.add_axes([pos.x0 + pos.width * 0.92, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('fft', labelpad=10)
    
    # Counts overlay in Bdot time range
    ax3_twin = ax3.twinx()
    # Adjust twin axis position
    ax3_twin.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    
    # Plot counts
    ax3_twin.plot(bin_centers, counts, 'w-', 
                    linewidth=2, alpha=0.7, label='Photon Counts')
    ax3_twin.set_xlim(tarr_B[0]*1e3, tarr_B[-1]*1e3)
    
    # Labels and formatting
    ax3.set_xlabel('Time (ms)', labelpad=10)
    ax3.set_ylabel('Frequency (MHz)', labelpad=10)
    ax3_twin.set_yticks([])
    
    # Add STFT information to title
    ax3.set_title('STFT and Photon Counts\n' +
                    f'Time Res: {time_resolution*1e3:.2f} ms, Freq Res: {freq_resolution/1e6:.2f} MHz',
                    pad=20)  # Add padding to title
    ax3_twin.legend(loc='upper right', bbox_to_anchor=(0.88, 1.0))
    
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

