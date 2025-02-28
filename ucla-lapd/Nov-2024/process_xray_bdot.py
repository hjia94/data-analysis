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

def process_shot(file_number, base_dir, monitor_idx=1):
    """Process a single shot and create a combined figure."""
    
    # Create figure with subplots - increased size and adjusted spacing
    fig = plt.figure(int(file_number), figsize=(14, 24))  # Wider and taller figure
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1.5], hspace=0.4)  # Increased vertical spacing
    ax1, ax2, ax3, ax4 = [fig.add_subplot(g) for g in gs]
    
    # Position window - increased scale for larger window
    select_monitor(monitor_idx=monitor_idx, window_scale=(0.4, 0.6))  # Wider and taller window

    # List all files in the directory
    all_files = os.listdir(base_dir)
    
    # Filter files for this shot number
    shot_files = [f for f in all_files if file_number in f]
    
    # Categorize files by type
    xray_files = [f for f in shot_files if "xray" in f.lower()]
    bdot_files = [f for f in shot_files if "Bdot" in f]
    # Power data files are those that don't contain "xray" or "Bdot"
    power_files = [f for f in shot_files if "xray" not in f.lower() and "Bdot" not in f]
    
    print(f"Found {len(xray_files)} X-ray files, {len(bdot_files)} Bdot files, and {len(power_files)} power files")
    
    # Plot 1: Power and Reflected Power (moved to first position)
    # Find current data (Channel 2)
    I_data = None
    tarr_I = None
    for f in power_files:
        if f"C2--" in f:
            filepath = os.path.join(base_dir, f)
            I_data, tarr_I = read_trc_data(filepath)
            print(f"Using current file: {f}")
            break
    
    # Find voltage data (Channel 4)
    V_data = None
    tarr_V = None
    for f in power_files:
        if f"C4--" in f:
            filepath = os.path.join(base_dir, f)
            V_data, tarr_V = read_trc_data(filepath)
            print(f"Using voltage file: {f}")
            break
    
    # Find reflected power data (Channel 3)
    Pref_data = None
    tarr_Pref = None
    for f in power_files:
        if f"C3--" in f:
            filepath = os.path.join(base_dir, f)
            Pref_data, tarr_Pref = read_trc_data(filepath)
            print(f"Using reflected power file: {f}")
            break
    
    # Plot power data if available
    if I_data is not None and V_data is not None and tarr_I is not None:
        # Calculate power: P = I * V (with scaling factor for current)
        P_data = (I_data * 0.25) * (-V_data)  # 1V ~ 0.25A; V is recorded negative
        
        # Create power plot with twin y-axis
        ax1.plot(tarr_I, P_data, 'b-', label='Power')
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Power (W)', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.grid(True)
        
        # Add reflected power if available
        if Pref_data is not None and tarr_Pref is not None:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(tarr_Pref, Pref_data, 'r-', label='Reflected Power')
            ax1_twin.set_ylabel('Reflected Power', color='r', fontsize=12)
            ax1_twin.tick_params(axis='y', labelcolor='r', labelsize=10)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        else:
            ax1.legend(loc='upper right', fontsize=10)
            
        ax1.set_title('Power and Reflected Power vs Time', fontsize=14, pad=10)
        
        # Free memory
        del P_data
        del I_data
        del V_data
        del tarr_I
        del tarr_V
    else:
        ax1.text(0.5, 0.5, 'Power data not available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Power Plot (Data Not Available)', fontsize=14, pad=10)
    
    # Find X-ray data (Channel 3)
    xray_data = None
    tarr_x = None
    
    # Look specifically for channel 3 for X-ray data
    for f in xray_files:
        if f"C3--" in f:
            filepath = os.path.join(base_dir, f)
            xray_data, tarr_x = read_trc_data(filepath)
            print(f"Using X-ray file: {f}")
            break

    if xray_data is None or tarr_x is None:
        raise FileNotFoundError(f"Required X-ray data files not found for file number {file_number}")
    
    # Invert signal for positive pulses
    xray_data = -xray_data
        
    # Process X-ray data
    detector = Photons(tarr_x, xray_data, savgol_window=31, distance_mult=0.0005,tsh_mult=[9, 150])
    detector.reduce_pulses()
    
    # Plot 2: Original signal and baseline
    ax2.plot(detector.tarr_ds, detector.data_ds, label='Original')
    ax2.plot(detector.tarr_ds, detector.baseline, label='Baseline')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Signal (V)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('Original Signal and Baseline', fontsize=14, pad=10)
    
    # Plot 3: Baseline-subtracted signal with pulses
    ax3.plot(detector.tarr_ds, detector.baseline_subtracted)
    ax3.axhline(y=detector.lower_threshold, color='g', linestyle='--', label='Lower Threshold')
    ax3.axhline(y=detector.upper_threshold, color='r', linestyle='--', label='Upper Threshold')
    ax3.scatter(detector.pulse_times, detector.pulse_amplitudes, color='red', label='Detected Pulses')
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Signal (V)', fontsize=12)
    ax3.tick_params(axis='both', labelsize=10)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_title('Baseline-subtracted Signal with Detected Pulses', fontsize=14, pad=10)
    
    # Get pulse data for counts
    pulse_times, pulse_areas = detector.get_pulse_arrays()
    
    # Calculate photon counts per bin
    bin_width_ms = 0.2
    bin_centers, counts = counts_per_bin(pulse_times, pulse_areas, bin_width_ms, amplitude_max=0.05)
    
    # Free memory
    del xray_data
    del tarr_x
    
    # Find and read Bdot data for all three channels
    bdot_data = {}  # Dictionary to store Bdot data for each channel
    
    # Look for Bdot data in channels 1, 2, and 3
    for channel in [1, 2, 3]:
        channel_key = f"C{channel}"
        found_file = False
        for f in bdot_files:
            if f"{channel_key}--" in f:
                filepath = os.path.join(base_dir, f)
                try:
                    data, tarr = read_trc_data(filepath)
                    # Verify data is valid
                    if len(data) > 0 and len(tarr) > 0:
                        bdot_data[channel_key] = {
                            'data': data,
                            'tarr': tarr,
                            'label': f"Bdot Channel {channel}"
                        }
                        print(f"Using Bdot file for {channel_key}: {f}")
                        found_file = True
                        break
                    else:
                        print(f"Warning: Empty data in Bdot file for {channel_key}: {f}")
                except Exception as e:
                    print(f"Error reading Bdot file for {channel_key}: {f} - {str(e)}")
        
        if not found_file:
            print(f"No valid Bdot file found for channel {channel_key}")
    
    # STFT plot (now ax4)
    if bdot_data:
        # Create a figure with subplots for the STFT plots
        num_channels = len(bdot_data)
        
        if num_channels > 0:
            # Define distinct colormaps for each channel
            colormaps = ['jet', 'viridis', 'plasma']
            # Define distinct alpha values that ensure visibility
            alpha_values = [0.9, 0.7, 0.5]
            # Get axis position once for consistent colorbar placement
            pos = ax4.get_position()
            
            # Process each channel's data and plot in the same axes
            for i, (channel_key, channel_data) in enumerate(bdot_data.items()):
                # Calculate STFT for this channel
                try:
                    freq_arr, fft_arr, _, _ = calculate_stft(
                        channel_data['tarr'], 
                        channel_data['data'],
                        samples_per_fft=500000, 
                        overlap_fraction=0.01, 
                        window='hanning', 
                        freq_min=150e6, 
                        freq_max=1000e6
                    )
                    
                    # Use a different colormap for each channel
                    cmap_index = min(i, len(colormaps)-1)
                    alpha = alpha_values[min(i, len(alpha_values)-1)]
                    
                    # Plot STFT
                    im = ax4.imshow(fft_arr.T, 
                           aspect='auto',
                           origin='lower',
                           extent=[channel_data['tarr'][0]*1e3, channel_data['tarr'][-1]*1e3, 
                                   freq_arr[0]/1e6, freq_arr[-1]/1e6],
                           interpolation='None',
                           cmap=colormaps[cmap_index],
                           alpha=alpha)
                    
                    # Add colorbar for each channel with consistent positioning
                    cax_width = 0.01
                    cax_spacing = 0.005
                    cax_height = (pos.height / num_channels) - cax_spacing
                    
                    # Position colorbars vertically stacked on the right
                    cax = fig.add_axes([
                        pos.x0 + pos.width * 0.92, 
                        pos.y0 + (num_channels - 1 - i) * (cax_height + cax_spacing), 
                        cax_width, 
                        cax_height
                    ])
                    
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label(channel_data['label'], fontsize=10)
                    
                    # Free memory immediately after plotting
                    del freq_arr
                    del fft_arr
                except Exception as e:
                    print(f"Error calculating STFT for {channel_key}: {str(e)}")
                    # Remove this channel from the dictionary so it's not included in the legend
                    bdot_data.pop(channel_key, None)
            
            # Add counts overlay on top of all STFT plots
            ax4_twin = ax4.twinx()
            ax4_twin.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            ax4_twin.plot(bin_centers, counts, 'w-', linewidth=2, alpha=1.0, label='Photon Counts')
            ax4_twin.set_yticks([])
            
            # Add a legend for the channels directly on the plot
            legend_elements = []
            for i, channel_key in enumerate(bdot_data.keys()):
                cmap_index = min(i, len(colormaps)-1)
                legend_elements.append(plt.Line2D([0], [0], color=plt.get_cmap(colormaps[cmap_index])(0.7), 
                                                 lw=4, label=f"Channel {channel_key}"))
            # Add photon counts to legend
            legend_elements.append(plt.Line2D([0], [0], color='white', lw=2, label='Photon Counts'))
            
            # Place legend in upper right corner with semi-transparent background
            ax4.legend(handles=legend_elements, loc='upper right', framealpha=0.7, fontsize=10)
            
            # Adjust STFT plot labels for better visibility
            ax4.set_xlabel('Time (ms)', fontsize=12)
            ax4.set_ylabel('Frequency (MHz)', fontsize=12)
            ax4.tick_params(axis='both', labelsize=10)
            ax4.set_title(f'STFT Spectrograms from {num_channels} Bdot Channels with Photon Counts', fontsize=14, pad=10)
            
            # Free memory
            for channel_key in bdot_data:
                if 'data' in bdot_data[channel_key]:
                    del bdot_data[channel_key]['data']
                if 'tarr' in bdot_data[channel_key]:
                    del bdot_data[channel_key]['tarr']
        else:
            ax4.text(0.5, 0.5, 'No Bdot data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('STFT Plot (Data Not Available)', fontsize=14, pad=10)
    else:
        ax4.text(0.5, 0.5, 'No Bdot data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('STFT Plot (Data Not Available)', fontsize=14, pad=10)
    
    # Add some padding around the entire figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    
    plt.pause(0.1)
    return fig

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":
    # Configuration
    file_numbers = [f"{i:05d}" for i in range(6,7)]  # Example: 00006
    base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t45_P24"
    
    print(f"Starting processing for {len(file_numbers)} shots")
    print(f"Data directory: {base_dir}")
    
    for file_number in file_numbers:
        try:
            process_shot(file_number, base_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Process terminated by KeyboardInterrupt")
            break

    print("\nScript execution completed")    
    plt.show()