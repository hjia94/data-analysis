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
from data_analysis_utils import Photons, calculate_stft, counts_per_bin
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

def process_shot(file_number, base_dir, debug=False):
    all_files = os.listdir(base_dir)
    bdot_files = [f for f in all_files if "Bdot" in f]

    for f in bdot_files:
        if "C1--" in f:
            filepath = os.path.join(base_dir, f)
            By_P21, tarr_B = read_trc_data(filepath)
            print(f"found By_P21 at {filepath}")
            break
    for f in bdot_files:
        if "C2--" in f:
            filepath = os.path.join(base_dir, f)
            Bx_P20, tarr_B = read_trc_data(filepath)
            print(f"found Bx_P20 at {filepath}")
            break
    for f in bdot_files:
        if "C3--" in f:
            filepath = os.path.join(base_dir, f)
            By_P20, tarr_B = read_trc_data(filepath)
            print(f"found By_P20 at {filepath}")
            break

    if By_P21 is None or Bx_P20 is None or By_P20 is None:
        raise FileNotFoundError(f"Required Bdot data files not found for file number {file_number}")
    


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
        threshold = [10, 1000]
        min_ts = 0.8e-6
        d = 0.2
    elif "p24" in f:
        threshold = [10, 150]
        min_ts = 1e-6
        d = 1
    elif "p30" in f:
        threshold = [10, 150]
        min_ts = 1e-6
        d = 1

    detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)
    detector.reduce_pulses()

    return detector.pulse_times, detector.pulse_amplitudes

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
            ax3.plot(tarr_I*1e3, P_data*1e-4, 'b-', linewidth=2)
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Power (kW)')
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
        
        # X-ray data
        if file_number in analysis_dict:
            pulse_tarr, pulse_amp = analysis_dict[file_number]
        else:
            pulse_tarr, pulse_amp = process_shot_2(file_number, base_dir, debug=debug)
            
            # Save new results
            analysis_dict[file_number] = (pulse_tarr, pulse_amp)
            try:
                np.save(analysis_file, analysis_dict)
                print(f"Added new analysis result for shot {file_number}")
            except Exception as e:
                print(f"Error saving analysis results: {e}")

        # Calculate counts per bin for ALL pulses (no amplitude filtering)
        bin_centers, counts = counts_per_bin(pulse_tarr, pulse_amp, bin_width=1)
        
        v = get_vel_freefall()  # velocity of ball at chamber center
        r_arr = get_pos_freefall(v, uw_start-t0+bin_centers*1e-3)  # position of ball at each time bin
        
        # Store ALL pulse data for later filtering during plotting
        shot_data = {
            'file_number': file_number,
            't0': t0,
            'uw_start': uw_start,
            'v': v,
            'pulse_tarr': pulse_tarr,
            'pulse_amp': pulse_amp,
            'bin_centers': bin_centers,
            'counts': counts,
            'r_arr': r_arr
        }
        
        all_scatter_data.append(shot_data)

    plot_combined_scatter(all_scatter_data)

    
def plot_combined_scatter(all_scatter_data):
    # Create combined scatter plot with 2 panels after processing all shots
    print("\nCreating combined X-ray counts plot with 2 amplitude ranges...")
    fig_combined, axes = plt.subplots(1, 2, figsize=(16, 6), num="Combined_X-ray_Counts")
    
    # Define panel titles for the 2 amplitude ranges
    panel_titles = [
        '0-50% Amplitude Range',
        '50-100% Amplitude Range'
    ]
    
    # Determine time axis range from magnetron power plot (matching individual shot plots)
    time_min, time_max = -5, 40  # Same range as used in individual power plots
    
    # Calculate global amplitude range across all shots for consistent thresholding
    all_pulse_amps = []
    for shot_data in all_scatter_data:
        all_pulse_amps.extend(shot_data['pulse_amp'])
    
    if len(all_pulse_amps) > 0:
        global_min = np.min(all_pulse_amps)
        global_max = np.max(all_pulse_amps) / 2
        global_range = global_max - global_min
        
        # Define 2 threshold ranges based on global amplitude range
        threshold_ranges = [
            (global_min, global_min + 0.50 * global_range),  # 0-50%
            (global_min + 0.50 * global_range, global_max)   # 50-100%
        ]
    else:
        threshold_ranges = [(0, 1), (1, 2)]  # Fallback ranges
    
    # Plot data for each amplitude range in separate panels with individual normalization
    scatter_list = []  # Store scatter objects for colorbar
    
    for panel_idx in range(2):
        ax = axes[panel_idx]
        min_thresh, max_thresh = threshold_ranges[panel_idx]
        
        # Collect data points for this amplitude range across all shots
        all_bin_centers = []
        all_r_positions = []
        all_counts = []
        
        for shot_data in all_scatter_data:
            # Filter pulses by amplitude range for this panel
            pulse_tarr = shot_data['pulse_tarr']
            pulse_amp = shot_data['pulse_amp']
            
            # Create mask for pulses in this amplitude range
            amp_mask = (pulse_amp >= min_thresh) & (pulse_amp <= max_thresh)
            filtered_pulse_tarr = pulse_tarr[amp_mask]
            filtered_pulse_amp = pulse_amp[amp_mask]
            
            if len(filtered_pulse_tarr) > 0:
                # Recalculate counts per bin for filtered pulses
                bin_centers, counts = counts_per_bin(filtered_pulse_tarr, filtered_pulse_amp, bin_width=1)
                
                if len(counts) > 0:
                    # Calculate radial positions for this shot
                    v = shot_data['v']
                    t0 = shot_data['t0']
                    uw_start = shot_data['uw_start']
                    r_arr = get_pos_freefall(v, uw_start-t0+bin_centers*1e-3)
                    
                    all_bin_centers.extend(bin_centers)
                    all_r_positions.extend(r_arr * 100)  # Convert to cm
                    all_counts.extend(counts)
        
        # Create scatter plot for this panel if there's data
        if len(all_counts) > 0:
            # Normalize counts for this panel to 0-1 range
            panel_max = np.max(all_counts)
            normalized_counts = np.array(all_counts) / panel_max if panel_max > 0 else np.array(all_counts)
            
            scatter = ax.scatter(all_bin_centers, all_r_positions, c=normalized_counts, 
                               s=50, alpha=0.7, vmin=0, vmax=1)
            scatter_list.append(scatter)
        else:
            # Create empty scatter for panels with no data
            scatter = ax.scatter([], [], c=[], s=50, alpha=0.7, vmin=0, vmax=1)
            scatter_list.append(scatter)
        
        # Set consistent time axis range to match magnetron power plots
        ax.set_xlim(time_min, time_max)
        ax.set_ylim(0,40)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Radial position (cm)')
        ax.grid(True)
    
    # Add a single colorbar for all panels with normalized scale
    plt.tight_layout()
    cbar = fig_combined.colorbar(scatter_list[0], ax=axes, label='normalized counts', shrink=0.8, aspect=30)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=18)
    
    plt.draw()
    plt.pause(0.1)

    # Keep the plots visible at the end
    plt.show(block=True)  # This will block until all figures are closed

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

    file_numbers = [f"{i:05d}" for i in range(6,25)]
    base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw17t47_P24"
    # base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw17t27_P30"
    # base_dir = r"E:\good_data\kapton\He3kA_B250G500G_pl0t20_uw15t35"


    # Uncomment one of these functions to run
    # main_plot(file_numbers, base_dir, debug=False)  # Process and display individual shots
    xray_wt_cam(file_numbers, base_dir, debug=False)
