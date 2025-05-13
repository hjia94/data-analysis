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

def process_shot(file_number, base_dir, bdot_channel=1, debug=False):
    """Process a single shot and create a combined figure."""
    
    # Create figure with 2 subplots using a string identifier
    fig_id = f"shot_{file_number}"
    fig, ax1 = plt.subplots(figsize=(8, 5), num=fig_id)
    
    # Position the window on the specified monitor
    # select_monitor(monitor_idx=monitor_idx, window_scale=(0.1, 0.1))
    
    # Create 2 subplots
    # gs = GridSpec(1, 1, figure=fig)
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])
    
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
    
    # Plot 1: Power and Reflected Power
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
    for f in power_files:
        if f"C3--" in f:
            filepath = os.path.join(base_dir, f)
            Pref_data, _ = read_trc_data(filepath)
            print(f"Using reflected power file: {f}")
            break
    
    # Plot power data if available
    if I_data is not None and V_data is not None and tarr_I is not None:
        # Calculate power: P = I * V (with scaling factor for current)
        P_data = (I_data * 0.25) * (-V_data)  # 1V ~ 0.25A; V is recorded negative
        
        # Create power plot with twin y-axis
        ax1.plot(tarr_I*1e3, P_data * 0.6e-3, label='Magnetron Power')
        ax1.set_xlabel('Time (ms)', fontsize=16)
        ax1.set_ylabel('Power (kW)', fontsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.grid(True)
        
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
    
    old_interactive = plt.isinteractive()
    plt.ion()  # Turn on interactive mode so figures display immediately

    # Parameters for P24 data
    # detector = Photons(tarr_x, xray_data, savgol_window=31, distance_mult=0.001, tsh_mult=[9, 150], debug=debug)
    detector = Photons(tarr_x, xray_data, min_timescale=1e-6, distance_mult=1, tsh_mult=[9, 150], debug=debug)
    detector.reduce_pulses()

    ax1_twin = ax1.twinx()
    ax1_twin.plot(detector.tarr_ds, detector.baseline_subtracted, c='r',label='X-ray signal')
    ax1_twin.set_yticks([])
    ax1.set_xlim(-5,40)
    ax1_twin.set_xlim(-5,40)
    
    # Calculate photon counts per bin
    bin_centers, counts = detector.counts_per_bin(bin_width_ms=0.1)
    
    # Free memory
    del xray_data
    del tarr_x
    
    # # Plot 2: STFT from Bdot data
    # # First try to load specified channel, then Channel 2 as fallback
    # bdot_data = None
    # tarr_B = None
    # channel_used = None
    
    # # Try to load the requested channel first
    # print(f"Attempting to load Bdot data from Channel {bdot_channel}")
    # for f in bdot_files:
    #     if f"C{bdot_channel}--" in f:
    #         filepath = os.path.join(base_dir, f)
    #         try:
    #             bdot_data, tarr_B = read_trc_data(filepath)
    #             channel_used = f"C{bdot_channel}"
    #             print(f"Using Bdot data from Channel {bdot_channel}: {f}")
    #             break
    #         except Exception as e:
    #             print(f"Error loading Bdot data from Channel {bdot_channel}: {e}")
                
    # # If requested channel failed, try channel 2 as fallback
    # if bdot_data is None or tarr_B is None:
    #     print("Falling back to Channel 2 for Bdot data")
    #     for f in bdot_files:
    #         if "C2--" in f:
    #             filepath = os.path.join(base_dir, f)
    #             try:
    #                 bdot_data, tarr_B = read_trc_data(filepath)
    #                 channel_used = "C2"
    #                 print(f"Using Bdot data from Channel 2 as fallback: {f}")
    #                 break
    #             except Exception as e:
    #                 print(f"Error loading Bdot data from Channel 2: {e}")
    
    # # Calculate and plot STFT if Bdot data is available
    # if bdot_data is not None and tarr_B is not None:
    #     try:
    #         # Calculate STFT using parameters from xray_test.ipynb
    #         freq, stft_matrix, stft_time = calculate_stft(
    #             time_array=tarr_B, 
    #             data_arr=bdot_data,
    #             freq_bins=500, 
    #             overlap_fraction=0.05,
    #             window='hanning', 
    #             freq_min=100e6, 
    #             freq_max=800e6
    #         )
            
    #         # Plot STFT with photon counts overlay
    #         plot_stft_wt_photon_counts(stft_time, stft_matrix, freq, bin_centers, counts, fig=fig, ax=ax2)
            
    #         # Adjust STFT plot labels for better visibility
    #         ax2.set_xlim(10,14)
            
    #     except Exception as e:
    #         print(f"Error calculating STFT: {e}")
    #         ax2.text(0.5, 0.5, f'Error calculating STFT: {str(e)}', 
    #                 horizontalalignment='center', verticalalignment='center',
    #                 transform=ax2.transAxes, fontsize=12)
    # else:
    #     ax2.text(0.5, 0.5, 'Bdot data not available', 
    #             horizontalalignment='center', verticalalignment='center',
    #             transform=ax2.transAxes, fontsize=14)
    #     ax2.set_title('STFT Plot (Data Not Available)', fontsize=14, pad=10)
    
    # Add some padding around the entire figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    
    # Draw the figure to ensure it shows up
    fig.canvas.draw()
    plt.pause(0.1)
    
    # Reset the interactive mode to what it was
    if not old_interactive:
        plt.ioff()
    
    # Get the lines and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()

    # Combine them
    lines = lines1 + lines2
    labels = labels1 + labels2

    # Create a single legend with all lines
    ax1.legend(lines, labels, loc='upper right', fontsize=16)

    return fig

def process_shot_2(file_number, base_dir, bdot_channel=1):
    """Process a single shot and return data for averaging."""
    
    all_files = os.listdir(base_dir)
    shot_files = [f for f in all_files if file_number in f]
    xray_files = [f for f in shot_files if "xray" in f.lower()]
    cam_files = [f for f in shot_files if f.endswith('.cine')]

    xray_data = None
    tarr_x = None
    for f in xray_files:
        if f"C3--" in f:
            filepath = os.path.join(base_dir, f)
            xray_data, tarr_x = read_trc_data(filepath)
            print(f"Using X-ray file: {f}")
            break

    if xray_data is None or tarr_x is None:
        raise FileNotFoundError(f"Required X-ray data files not found for file number {file_number}")

    detector = Photons(tarr_x, xray_data, min_timescale=1e-6, distance_mult=1, tsh_mult=[9, 150], debug=False)        
    detector.reduce_pulses()

    min_threshold = 0.05
    max_threshold = 0.3
    t_bin = 1
    bin_centers, counts = detector.counts_per_bin(t_bin, amplitude_min=min_threshold, amplitude_max=max_threshold)

    # Free memory
    del xray_data
    del tarr_x
    return bin_centers, counts

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

def main_plot(file_numbers, base_dir, debug):
    
    print(f"Starting processing for {len(file_numbers)} shots")
    print(f"Data directory: {base_dir}")
    
    # Turn on interactive mode for the whole script
    plt.ion()
    
    # List to store figures so they don't get garbage collected
    all_figures = []
    
    for file_number in file_numbers:
        try:
            fig = process_shot(file_number, base_dir, bdot_channel=2, debug=debug)
            all_figures.append(fig)
            
            # Add explicit pause to ensure the display updates
            plt.pause(0.5)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("Process terminated by KeyboardInterrupt")
            break

    print("\nScript execution completed")
    
    # Create a small tkinter window to keep the plots alive without blocking
    # This helps prevent figures from closing when script ends
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    # Turn off interactive mode and show all figures
    plt.ioff()
    plt.show(block=True)  # Block until all figures are closed

def xray_wt_cam(file_numbers, base_dir, debug=False):

    for file_number in file_numbers:
        bin_centers, counts = process_shot_2(file_number, base_dir)

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

    file_numbers = [f"{i:05d}" for i in range(11,15)]
    # base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t45_P24"
    base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30"

    # Uncomment one of these functions to run
    main_plot(file_numbers, base_dir, debug=False)  # Process and display individual shots
