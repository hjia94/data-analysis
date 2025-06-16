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
from matplotlib import colors
from screeninfo import get_monitors
import tkinter as tk
import cv2
import re
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
# Add paths for custom modules
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\object_tracking")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")


from read_scope_data import read_trc_data
from data_analysis_utils import Photons, calculate_stft
from plot_utils import select_monitor, plot_stft_wt_photon_counts, plot_original_and_baseline, plot_subtracted_signal
from read_cine import read_cine, convert_cine_to_avi
from track_object import track_object, detect_chamber, get_vel_freefall, get_pos_freefall

#===========================================================================================================
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

#===========================================================================================================

def get_magnetron_power_data(power_files, base_dir):
    """
    Calculate magnetron power data from oscilloscope files.
    
    Parameters:
    -----------
    power_files : list
        List of power data filenames
    base_dir : str
        Base directory path containing the data files
        
    Returns:
    --------
    tuple
        (P_data, tarr_I, I_data, V_data, Pref_data) where:
        - P_data: calculated power array
        - tarr_I: time array for current data
        - I_data: current data array
        - V_data: voltage data array
        - Pref_data: reflected power data array (or None if not available)
    """
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
    
    # Calculate power if both current and voltage data are available
    P_data = None
    if I_data is not None and V_data is not None and tarr_I is not None:
        # Calculate power: P = I * V (with scaling factor for current)
        P_data = (I_data * 0.25) * (-V_data) * 0.6  # 1V ~ 0.25A; V is recorded negative; assume 60% of power goes to the plasma
    
    return P_data, tarr_I, I_data, V_data, Pref_data

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
    # Get magnetron power data using the standalone function
    P_data, tarr_I, I_data, V_data, Pref_data = get_magnetron_power_data(power_files, base_dir)
    
    # Plot power data if available
    if P_data is not None and tarr_I is not None:
        # Create power plot with twin y-axis
        ax1.plot(tarr_I*1e3, P_data * 1e-3, label='Magnetron Power')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Power (kW)')
        ax1.tick_params(axis='y')
        ax1.tick_params(axis='x')
        ax1.grid(True)
        
        # Free memory
        del P_data
        del I_data
        del V_data
        del tarr_I
    else:
        ax1.text(0.5, 0.5, 'Power data not available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes)
        ax1.set_title('Power Plot (Data Not Available)', pad=10)
    
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

    if "p24" in f:
        detector = Photons(tarr_x, xray_data, savgol_window=31, distance_mult=0.001, tsh_mult=[9, 150], debug=debug)
    elif "p30" in f:
        detector = Photons(tarr_x, xray_data, min_timescale=1e-6, distance_mult=1, tsh_mult=[1, 300], debug=debug)
    else: print('Huh?')
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
    #                 transform=ax2.transAxes)
    # else:
    #     ax2.text(0.5, 0.5, 'Bdot data not available', 
    #             horizontalalignment='center', verticalalignment='center',
    #             transform=ax2.transAxes)
    #     ax2.set_title('STFT Plot (Data Not Available)', pad=10)
    
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
    ax1.legend(lines, labels, loc='upper right')

    return fig

def process_shot_2(file_number, base_dir, debug=False):
    """Process a single shot and return data for averaging."""
    
    all_files = os.listdir(base_dir)
    shot_files = [f for f in all_files if file_number in f]
    xray_files = [f for f in shot_files if "xray" in f.lower()]
    
    xray_data = None
    tarr_x = None

    for f in xray_files:
        if "C3--" in f:
            filepath = os.path.join(base_dir, f)
            xray_data, tarr_x = read_trc_data(filepath)
            print(f"Using X-ray file: {f}")
            break

    if xray_data is None or tarr_x is None:
        raise FileNotFoundError(f"Required X-ray data files not found for file number {file_number}")

    if 'kapton' in f:
        threshold = [20, 1000]
        min_threshold = 0.01
        max_threshold = 0.5
    elif "p24" in f:
        threshold = [9, 150]
        min_threshold = 0.025
        max_threshold = 0.3
    elif "p30" in f:
        threshold = [1, 300]
        min_threshold = 0.015
        max_threshold = 0.3

    min_ts = 1e-6
    d = 1
    detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)

    detector.reduce_pulses()

    t_bin = 1
    bin_centers, counts = detector.counts_per_bin(t_bin, amplitude_min=min_threshold, amplitude_max=max_threshold)

    # Free memory
    del xray_data
    del tarr_x
    return bin_centers, counts

def process_video(file_number, base_dir, fig=None, ax=None):
    '''
    Process the video file for a given shot number and return the time at which the ball reaches the chamber center
    '''
    all_files = os.listdir(base_dir)
    actual_number = int(file_number)
    cam_files = [f for f in all_files if f.endswith('.cine') and 
                 int(f.split('_')[-1].replace('.cine', '')) == actual_number]

    if not cam_files:
        raise FileNotFoundError(f"No video file found for shot number {file_number}")

    filepath = os.path.join(base_dir, cam_files[0])
    avi_path = filepath.replace('.cine', '.avi')
    print(f"Using video file: {filepath}")

    # Define path for tracking results
    tracking_file = os.path.join(base_dir, 'tracking_results.npy')
    
    # Initialize or load the tracking dictionary
    tracking_dict = {}
    if os.path.exists(tracking_file):
        try:
            tracking_dict = np.load(tracking_file, allow_pickle=True).item()
            print(f"Loaded {len(tracking_dict)} existing tracking results")
        except Exception as e:
            print(f"Error loading tracking results: {e}")
            tracking_dict = {}

    # Check if we already have results for this file
    if filepath in tracking_dict:
        print(f"Loading existing tracking results for {filepath}")
        cf, ct = tracking_dict[filepath]
    else:
        print(f"No existing tracking results found. Processing video...")
        tarr, frarr, dt = read_cine(filepath)

        if not os.path.exists(avi_path):
            print(f"Converting {filepath} to {avi_path}")
            convert_cine_to_avi(frarr, avi_path)

        print(f"Tracking object in {avi_path}")
        parr, frarr, cf = track_object(avi_path)
        ct = tarr[cf]
        
        # Add new result to the dictionary and save
        tracking_dict[filepath] = (cf, ct)
        try:
            np.save(tracking_file, tracking_dict)
            print(f"Added new tracking result and saved to {tracking_file}")
        except Exception as e:
            print(f"Error saving tracking results: {e}")

    # Plot frame with detected object at center of chamber
    cap = cv2.VideoCapture(avi_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cf)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame")

    # Detect chamber
    (cx, cy), chamber_radius = detect_chamber(frame)
    
    # Plot in the provided axis
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    chamber_circle = plt.Circle((cx, cy), chamber_radius, fill=False, color='green', linewidth=2)
    ax.add_patch(chamber_circle)
    print(f"ball reaches chamber center at t={ct * 1e3:.3f}ms from plasma trigger")
    ax.axis('off')

    cap.release()
    return ct, filepath

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
    # Turn on interactive mode for the whole script
    plt.ion()
    
    # Define path for analysis results
    analysis_file = os.path.join(base_dir, 'analysis_results.npy')
    
    # Initialize or load the analysis dictionary
    analysis_dict = {}
    if os.path.exists(analysis_file):
        try:
            analysis_dict = np.load(analysis_file, allow_pickle=True).item()
            print(f"Loaded {len(analysis_dict)} existing analysis results")
        except Exception as e:
            print(f"Error loading analysis results: {e}")
            analysis_dict = {}
    
    # Storage for combined scatter plot data
    all_scatter_data = []
    max_counts = 0
    
    # Process each file: collect data and create individual figures
    print("Processing each shot...")
    for file_number in file_numbers:
        print(f"\nProcessing shot {file_number}")

        
        # Create individual figure with two subplots (video + power)
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5), num=f"shot_{file_number}")
        

        t0, filepath = process_video(file_number, base_dir, fig=fig, ax=ax1)
        match = re.search(r'uw(\d+)t', filepath)
        uw_start = int(match.group(1)) * 1e-3
        print(f"uw_start: {uw_start}")

              
        # Plot magnetron power in second panel
        all_files = os.listdir(base_dir)
        shot_files = [f for f in all_files if file_number in f]
        power_files = [f for f in shot_files if "xray" not in f.lower() and "Bdot" not in f]
        
        # Get magnetron power data using the standalone function
        P_data, tarr_I, I_data, V_data, Pref_data = get_magnetron_power_data(power_files, base_dir)
        
        # Plot power data if available
        if P_data is not None and tarr_I is not None:
            ax3.plot(tarr_I*1e3, P_data, 'b-', linewidth=2)
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Power (kW)')
            ax3.set_title('Magnetron Power')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'Power data not available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes)
            ax3.set_title('Magnetron Power (Data Not Available)')
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Power (kW)')
        
        # Adjust layout and display for individual figure
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Small pause to ensure the plot is rendered
        
        # Collect data for combined scatter plot
        if file_number in analysis_dict:
            bin_centers, counts, r_arr = analysis_dict[file_number]
        else:
            # Get x-ray data
            bin_centers, counts = process_shot_2(file_number, base_dir, debug=debug)
            v = get_vel_freefall()  # velocity of ball at chamber center
            r_arr = get_pos_freefall(v, uw_start-t0+bin_centers*1e-3)  # position of ball at each time bin
            
            # Save new results
            analysis_dict[file_number] = (bin_centers, counts, r_arr)
            try:
                np.save(analysis_file, analysis_dict)
                print(f"Added new analysis result for shot {file_number}")
            except Exception as e:
                print(f"Error saving analysis results: {e}")
        
        # Store data for combined plot
        all_scatter_data.append({
            'file_number': file_number,
            'bin_centers': bin_centers,
            'counts': counts,
            'r_arr': r_arr,
            't0': t0,
            'uw_start': uw_start
        })
        
        # Update maximum counts
        max_counts = max(max_counts, np.max(counts))
    
    # Create combined scatter plot after processing all shots
    print("\nCreating combined X-ray counts plot...")
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8), num="Combined_X-ray_Counts")

    
    for i, data in enumerate(all_scatter_data):
        scatter = ax_combined.scatter(data['bin_centers'], data['r_arr']*100, c=data['counts'], s=50, alpha=0.7)
        # norm=colors.PowerNorm(gamma=0.5, vmin=0, vmax=max_counts)
        
    ax_combined.set_xlabel('Time (ms)')
    ax_combined.set_ylabel('Radial position (cm)')
    ax_combined.grid(True)
    
    # Add colorbar to combined plot
    cbar = plt.colorbar(scatter, ax=ax_combined, label='Counts')
    cbar.ax.tick_params(labelsize=18)
    
    plt.draw()
    plt.pause(0.1)

    # Keep the plots visible at the end
    plt.show(block=True)  # This will block until all figures are closed

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

    file_numbers = [f"{i:05d}" for i in range(30,40)]
    # base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw17t47_P24"
    # base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw17t27_P30"
    base_dir = r"E:\good_data\kapton\He3kA_B380G800G_pl0t20_uw15t35"

    # Uncomment one of these functions to run
    # main_plot(file_numbers, base_dir, debug=False)  # Process and display individual shots
    xray_wt_cam(file_numbers, base_dir, debug=False)
