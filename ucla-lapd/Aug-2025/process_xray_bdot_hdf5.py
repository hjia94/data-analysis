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
from scipy import ndimage
import tkinter as tk
import cv2
import re
import h5py
import glob


# Dynamically add all subdirectories under the parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for subdir in glob.glob(os.path.join(parent_dir, '**'), recursive=True):
    if os.path.isdir(subdir):
        sys.path.append(subdir)


from read_scope_data import read_trc_data, read_hdf5_scope_data, read_hdf5_scope_tarr
from data_analysis_utils import Photons, calculate_stft, counts_per_bin
from plot_utils import select_monitor, plot_stft_wt_photon_counts, plot_original_and_baseline, plot_subtracted_signal
from read_cine import read_cine, convert_cine_to_avi
from track_object import track_object, detect_chamber, get_vel_freefall, get_pos_freefall

#===========================================================================================================
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

#===========================================================================================================

def get_magnetron_power_data(hdf5_filename, shot_number=1, scope_name='magscope'):
    """
    Calculate magnetron power from HDF5 file data.
    
    Parameters:
    -----------
    hdf5_filename : str
        Path to the HDF5 file
    shot_number : int
        Shot number to process (default: 1)
    scope_name : str
        Name of the scope group to use (default: 'magscope')
        
    Returns:
    --------
    tuple: (P_data, tarr, I_data, V_data, Pref_data)
        P_data : Power calculation (W)
        tarr : Time array (s)
        I_data : Current data
        V_data : Voltage data
        Pref_data : Reflected power data
    """
    # Initialize return variables
    I_data = None
    V_data = None
    Pref_data = None
    I_channel = None
    V_channel = None
    Pref_channel = None
    
    # Read time array
    try:
        tarr = read_hdf5_scope_tarr(hdf5_filename, scope_name)
    except Exception as e:
        print(f"Error reading time array: {e}")
        return None, None, None, None, None
    
    # Find the appropriate channels based on descriptions
    with h5py.File(hdf5_filename, 'r') as f:
        # Check if scope exists
        if scope_name not in f:
            print(f"Scope '{scope_name}' not found in file.")
            return None, tarr, None, None, None
            
        scope_group = f[scope_name]
        
        # Find the first non-skipped shot for channel descriptions
        shot_groups = [k for k in scope_group.keys() if k.startswith('shot_')]
        if not shot_groups:
            print("No shot data found in this scope group.")
            return None, tarr, None, None, None
            
        # Get first available shot for channel descriptions
        shot_group = None
        for shot_name in shot_groups:
            current_shot = scope_group[shot_name]
            if 'skipped' not in current_shot.attrs or not current_shot.attrs['skipped']:
                shot_group = current_shot
                break
                
        if shot_group is None:
            print("All shots were skipped in this scope group.")
            return None, tarr, None, None, None
            
        # Find channels based on descriptions
        channel_names = [k.split('_')[0] for k in shot_group.keys() if k.endswith('_data')]
        
        for channel in channel_names:
            data_key = f"{channel}_data"
            if data_key in shot_group:
                if 'description' in shot_group[data_key].attrs:
                    desc = shot_group[data_key].attrs['description'].lower()
                    if 'magnetron current' in desc:
                        I_channel = channel
                        print(f"Found current channel: {channel} - {shot_group[data_key].attrs['description']}")
                    elif 'magnetron voltage' in desc:
                        V_channel = channel
                        print(f"Found voltage channel: {channel} - {shot_group[data_key].attrs['description']}")
                    elif 'reflected power' in desc:
                        Pref_channel = channel
                        print(f"Found reflected power channel: {channel} - {shot_group[data_key].attrs['description']}")
    
    # Read data for the requested shot number
    if I_channel is not None:
        try:
            I_data = read_hdf5_scope_data(hdf5_filename, scope_name, I_channel, shot_number)
            
            # Apply scaling if specified in description
            with h5py.File(hdf5_filename, 'r') as f:
                shot_group = f[scope_name][f'shot_{shot_number}']
                desc = shot_group[f'{I_channel}_data'].attrs['description'].lower()
                
                # Extract scaling factor if present (e.g., "0.25A/V")
                import re
                match = re.search(r'(\d+\.?\d*)a/v', desc)
                if match:
                    scale_factor = float(match.group(1))
                    print(f"Applying current scaling factor: {scale_factor} A/V")
                    I_data = I_data * scale_factor
        except Exception as e:
            print(f"Error reading current data: {e}")
            I_data = None
    
    if V_channel is not None:
        try:
            V_data = read_hdf5_scope_data(hdf5_filename, scope_name, V_channel, shot_number)
        except Exception as e:
            print(f"Error reading voltage data: {e}")
            V_data = None
    
    if Pref_channel is not None:
        try:
            Pref_data = read_hdf5_scope_data(hdf5_filename, scope_name, Pref_channel, shot_number)
        except Exception as e:
            print(f"Error reading reflected power data: {e}")
            Pref_data = None
    
    # Calculate power if both current and voltage data are available
    P_data = None
    if I_data is not None and V_data is not None:
        # Calculate power: P = I * V
        # Voltage is typically recorded negative in these experiments
        # Also assuming 60% efficiency factor (power delivered to plasma)
        # Apply Gaussian filter to smooth output power
        P_data = I_data * (-V_data) * 0.6
        P_data = ndimage.gaussian_filter1d(P_data, sigma=100)
        print(f"Magnetron power calculated for shot {shot_number}")
    else:
        print("Cannot calculate power: missing current or voltage data")
    
    return P_data, tarr, I_data, V_data, Pref_data

def process_shot_bdot(file_number, base_dir, debug=False):
    '''
    Process a single shot of bdot data and return data for averaging.
    '''
    all_files = os.listdir(base_dir)
    shot_files = [f for f in all_files if file_number in f]
    bdot_files = [f for f in shot_files if "Bdot" in f]

    # Check if any Bdot files exist for this shot
    if not bdot_files:
        raise FileNotFoundError(f"No Bdot files found for file number {file_number}")

    By_P21 = None
    Bx_P20 = None
    By_P20 = None
    tarr_B = None

    for f in bdot_files:
        if "C1--" in f:
            filepath = os.path.join(base_dir, f)
            By_P21, tarr_B = read_trc_data(filepath)
            print(f"found By_P21 at {filepath}")

        if "C2--" in f:
            filepath = os.path.join(base_dir, f)
            Bx_P20, tarr_B = read_trc_data(filepath)
            print(f"found Bx_P20 at {filepath}")

        if "C3--" in f:
            filepath = os.path.join(base_dir, f)
            By_P20, tarr_B = read_trc_data(filepath)
            print(f"found By_P20 at {filepath}")


    # Calculate STFT for each Bdot signal
    freq_bins = 1000
    overlap_fraction = 0.05
    freq_min = 200e6  # 100 MHz
    freq_max = 2000e6  # 800 MHz

    if By_P21 is not None:
        freq, stft_matrix1, stft_time = calculate_stft(tarr_B, By_P21, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
    else:
        stft_matrix1 = None

    if Bx_P20 is not None:
        freq, stft_matrix2, stft_time = calculate_stft(tarr_B, Bx_P20, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
    else:
        stft_matrix2 = None

    if By_P20 is not None:
        freq, stft_matrix3, stft_time = calculate_stft(tarr_B, By_P20, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
    else:
        stft_matrix3 = None

    return stft_time, freq, stft_matrix1, stft_matrix2, stft_matrix3

def process_shot_xray(file_number, base_dir, debug=False):
    """Process a single shot of xray data and return data for averaging."""
    
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

    if "380G800G" in base_dir:
        d = 0.1
        if "kapton" in base_dir:
            threshold = [8, 400]
            min_ts = 0.8e-6
        else:
            threshold = [20, 150]
            min_ts = 1e-6
            
    elif "P24" and "250G500G"in base_dir:
        threshold = [10, 150]
        min_ts = 1e-6
        d = 1
    elif "P30" and "250G500G" in base_dir:
        threshold = [10, 150]
        min_ts = 1e-6
        d = 1
    elif "500G1kG" in base_dir:
        threshold = [20, 250]
        min_ts = 1e-6
        d = 0.1

    tarr_x = tarr_x #[:int(len(tarr_x)/2)]
    xray_data = xray_data #[:int(len(xray_data)/2)]

    detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)
    detector.reduce_pulses()

    return detector.pulse_times, detector.pulse_amplitudes

def process_video(file_number, base_dir):
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

    cap.release()
    return ct, frame, (cx, cy), chamber_radius, filepath

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

def xray_wt_cam(file_numbers, base_dir, plot_Bdot=False, debug=False):
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
    
    # Storage for STFT data averaging
    all_stft_matrix1 = []
    all_stft_matrix2 = []
    all_stft_matrix3 = []
    stft_time_ref = None
    freq_ref = None
    
    # Process each file: collect data and create individual figures
    print("Processing each shot...")
    for file_number in file_numbers:
        print(f"\nProcessing shot {file_number}")
        # Video data
        try:
            t0, frame, (cx, cy), chamber_radius, filepath = process_video(file_number, base_dir)
        except FileNotFoundError as e:
            print(f"No video file found for shot {file_number}; skipping...")
            continue
        
        # Magnetron power data
        all_files = os.listdir(base_dir)
        shot_files = [f for f in all_files if file_number in f]
        power_files = [f for f in shot_files if "xray" not in f.lower() and "Bdot" not in f]
        
        # Get magnetron power data using the standalone function
        P_data, tarr_I, I_data, V_data, Pref_data = get_magnetron_power_data(power_files, base_dir)

        # Microwave start time
        match = re.search(r'uw(\d+)t', filepath)
        uw_start = int(match.group(1)) * 1e-3
        if debug:
            print(f"uw_start: {uw_start}")

        
        # Create individual figure with two subplots (video + power)
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5), num=f"shot_{file_number}")

        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        chamber_circle = plt.Circle((cx, cy), chamber_radius, fill=False, color='green', linewidth=2)
        ax1.add_patch(chamber_circle)
        print(f"ball reaches chamber center at t={t0 * 1e3:.3f}ms from plasma trigger")
        ax1.axis('off')

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
        plt.pause(0.1)
        
        
        # X-ray data
        if file_number in analysis_dict:
            pulse_tarr, pulse_amp = analysis_dict[file_number]
        else:
            pulse_tarr, pulse_amp = process_shot_xray(file_number, base_dir, debug=debug)
            
            # Save new results
            analysis_dict[file_number] = (pulse_tarr, pulse_amp)
            try:
                np.save(analysis_file, analysis_dict)
                print(f"Added new analysis result for shot {file_number}")
            except Exception as e:
                print(f"Error saving analysis results: {e}")
        
        # Store ALL pulse data for plot in the end
        shot_data = {
            'file_number': file_number,
            't0': t0,
            'uw_start': uw_start,
            'pulse_tarr': pulse_tarr,
            'pulse_amp': pulse_amp,
        }
        
        all_scatter_data.append(shot_data)

    
        if plot_Bdot:
            try:
                stft_tarr, freq_arr, stft_matrix1, stft_matrix2, stft_matrix3 = process_shot_bdot(file_number, base_dir, debug=debug)
            except FileNotFoundError as e:
                print(f"No Bdot data found for shot {file_number}; skipping...")
                continue
            
            # Collect STFT matrices for averaging
            if stft_matrix1 is not None:
                all_stft_matrix1.append(stft_matrix1)
            if stft_matrix2 is not None:
                all_stft_matrix2.append(stft_matrix2)
            if stft_matrix3 is not None:
                all_stft_matrix3.append(stft_matrix3)

            # Average the STFT matrices with optional time axis limitation
            avg_stft_matrix1 = None
            avg_stft_matrix2 = None
            avg_stft_matrix3 = None
            
            # Option to limit time axis to first half (set to True to enable)
            limit_time_axis = False
            
            if len(all_stft_matrix1) > 0:
                matrices = np.array(all_stft_matrix1)
                if limit_time_axis:
                    index = matrices.shape[1] // 2
                    matrices = matrices[:, :index, :]
                avg_stft_matrix1 = np.mean(matrices, axis=0)
                print(f"Averaged {len(all_stft_matrix1)} By_P21 STFT matrices")
            
            if len(all_stft_matrix2) > 0:
                matrices = np.array(all_stft_matrix2)
                if limit_time_axis:
                    index = matrices.shape[1] // 2
                    matrices = matrices[:, :index, :]
                avg_stft_matrix2 = np.mean(matrices, axis=0)
                print(f"Averaged {len(all_stft_matrix2)} Bx_P20 STFT matrices")
            
            if len(all_stft_matrix3) > 0:
                matrices = np.array(all_stft_matrix3)
                if limit_time_axis:
                    index = matrices.shape[1] // 2
                    matrices = matrices[:, :index, :]
                avg_stft_matrix3 = np.mean(matrices, axis=0)
                print(f"Averaged {len(all_stft_matrix3)} By_P20 STFT matrices")
            
            if limit_time_axis:
                stft_tarr = stft_tarr[:index]
                freq_arr = freq_arr[:index]

            plot_averaged_bdot_stft(avg_stft_matrix1, avg_stft_matrix2, avg_stft_matrix3, stft_tarr, freq_arr)
    
    
    plot_combined_scatter(all_scatter_data, amplitude_ranges=None) # [(0, 0.5), (0.5, 1.0)])

    plt.show(block=True)
    

def plot_combined_scatter(all_scatter_data, amplitude_ranges=None):
    """
    Plot combined X-ray counts scatter plot with configurable amplitude ranges.
    
    Parameters:
    -----------
    all_scatter_data : list
        List of shot data dictionaries
    amplitude_ranges : list of tuples, default=None
        List of (start_fraction, end_fraction) tuples defining amplitude ranges as fractions of total range.
        If None, creates a single plot with all counts combined.
    """
    
    def process_shot_data(shot_data, min_thresh=None, max_thresh=None):
        """Extract and process data from a single shot with optional amplitude filtering."""
        pulse_tarr, pulse_amp = shot_data['pulse_tarr'], shot_data['pulse_amp']
        
        if len(pulse_tarr) == 0:
            return [], [], []
            
        # Apply amplitude filtering if thresholds provided
        if min_thresh is not None and max_thresh is not None:
            mask = (pulse_amp >= min_thresh) & (pulse_amp <= max_thresh)
            pulse_tarr, pulse_amp = pulse_tarr[mask], pulse_amp[mask]
            
        if len(pulse_tarr) == 0:
            return [], [], []
            
        # Calculate counts and positions
        bin_centers, counts = counts_per_bin(pulse_tarr, pulse_amp, bin_width=1)
        if len(counts) == 0:
            return [], [], []
            
        r_arr = get_pos_freefall(bin_centers*1e-3 + shot_data['uw_start'], shot_data['t0'])
        return bin_centers, (r_arr * 100), counts  # Convert to cm
    
    def create_scatter_plot(ax, bin_centers, r_positions, counts, vmax, add_colorbar=False):
        """Create a scatter plot with consistent formatting."""
        if len(counts) > 0:
            scatter = ax.scatter(bin_centers, r_positions, c=counts, s=50, alpha=0.7, 
                               vmin=0, vmax=vmax, cmap='viridis')
            if add_colorbar:
                cbar = plt.colorbar(scatter, ax=ax, label='counts', shrink=0.8)
                cbar.ax.tick_params(labelsize=18)
            return scatter
        return ax.scatter([], [], c=[], s=50, alpha=0.7, vmin=0, vmax=vmax, cmap='viridis')
    
    # Handle single plot case
    if amplitude_ranges is None:
        print("\nCreating single combined X-ray counts plot with all data...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Collect all data
        all_bin_centers, all_r_positions, all_counts = [], [], []
        for shot_data in all_scatter_data:
            bin_centers, r_positions, counts = process_shot_data(shot_data)
            all_bin_centers.extend(bin_centers)
            all_r_positions.extend(r_positions)
            all_counts.extend(counts)
        
        # Create plot
        max_counts = np.max(all_counts) if all_counts else 1
        create_scatter_plot(ax, all_bin_centers, all_r_positions, all_counts, max_counts, add_colorbar=True)
        
        ax.set_ylim(-40, 40)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Position (cm)')
        ax.grid(True)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        return
    
    # Multi-panel case
    if len(amplitude_ranges) == 0:
        amplitude_ranges = [(0, 0.25), (0.25, 0.5), (0.5, 1.0)]
    
    num_ranges = len(amplitude_ranges)
    print(f"\nCreating combined X-ray counts plot with {num_ranges} amplitude ranges...")
    
    # Setup figure
    fig_width = max(12, num_ranges * 5)
    fig, axes = plt.subplots(1, num_ranges, figsize=(fig_width, 6), sharey=True)
    if num_ranges == 1:
        axes = [axes]
    
    # Calculate amplitude thresholds
    all_pulse_amps = [amp for shot_data in all_scatter_data for amp in shot_data['pulse_amp']]
    if all_pulse_amps:
        global_min, global_max = np.min(all_pulse_amps), np.max(all_pulse_amps) / 2
        global_range = global_max - global_min
        threshold_ranges = [(global_min + start * global_range, global_min + end * global_range) 
                           for start, end in amplitude_ranges]
    else:
        threshold_ranges = [(i, i+1) for i in range(num_ranges)]
    
    # Collect data for all panels and find global max
    panel_data_list = []
    global_max_counts = 0
    
    for min_thresh, max_thresh in threshold_ranges:
        all_bin_centers, all_r_positions, all_counts = [], [], []
        for shot_data in all_scatter_data:
            bin_centers, r_positions, counts = process_shot_data(shot_data, min_thresh, max_thresh)
            all_bin_centers.extend(bin_centers)
            all_r_positions.extend(r_positions)
            all_counts.extend(counts)
        
        panel_data_list.append((all_bin_centers, all_r_positions, all_counts))
        if all_counts:
            global_max_counts = max(global_max_counts, np.max(all_counts))
    
    print(f"Global maximum counts across all panels: {global_max_counts}")
    
    # Create plots
    scatter_list = []
    for i, (ax, (bin_centers, r_positions, counts)) in enumerate(zip(axes, panel_data_list)):
        scatter = create_scatter_plot(ax, bin_centers, r_positions, counts, global_max_counts)
        scatter_list.append(scatter)
        
        ax.set_ylim(-40, 40)
        ax.set_xlabel('Time (ms)')
        if i == 0:
            ax.set_ylabel('Position (cm)')
        ax.grid(True)
    
    # Add shared colorbar
    plt.tight_layout()
    reference_scatter = next((s for s in scatter_list if s.get_array().size > 0), None)
    if reference_scatter:
        cbar = fig.colorbar(reference_scatter, ax=axes, label='counts', shrink=0.8, aspect=30)
        cbar.ax.tick_params(labelsize=18)
    
    plt.draw()
    plt.pause(0.1)

def plot_averaged_bdot_stft(avg_stft_matrix1, avg_stft_matrix2, avg_stft_matrix3, stft_tarr, freq_arr):
    """
    Plot averaged STFT matrices for the three Bdot signals.
    """
    print("\nCreating averaged Bdot STFT plots...")
    
    # Create figure with 3 subplots (similar to notebook)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), num="Averaged_Bdot_STFT", sharex=True)
    
    # Plot averaged STFT for By_P21 with log scale
    if avg_stft_matrix1 is not None:
        im1 = ax1.imshow(avg_stft_matrix1.T,
                         aspect='auto',
                         origin='lower',
                         extent=[stft_tarr[0]*1e3, stft_tarr[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
                         interpolation='None',
                         cmap='jet',
                         norm=colors.LogNorm(vmin=avg_stft_matrix1.T[avg_stft_matrix1.T > 0].min(), 
                                           vmax=avg_stft_matrix1.T.max()))
        ax1.set_ylabel('Frequency (MHz)')
        ax1.text(0.98, 0.95, 'By_P21 (x=8cm, y=0cm)', transform=ax1.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
                 horizontalalignment='right', verticalalignment='top')
        fig.colorbar(im1, ax=ax1, label='Magnitude')
    
    if avg_stft_matrix2 is not None:
        im2 = ax2.imshow(avg_stft_matrix2.T,
                         aspect='auto',
                         origin='lower',
                         extent=[stft_tarr[0]*1e3, stft_tarr[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
                         interpolation='None',
                         cmap='jet',
                         norm=colors.LogNorm(vmin=avg_stft_matrix2.T[avg_stft_matrix2.T > 0].min(), 
                                           vmax=avg_stft_matrix2.T.max()))
        ax2.set_ylabel('Frequency (MHz)')
        ax2.text(0.98, 0.95, 'Bx_P20 (x=0cm, y=0cm)', transform=ax2.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
                 horizontalalignment='right', verticalalignment='top')
        fig.colorbar(im2, ax=ax2, label='Magnitude')

    if avg_stft_matrix3 is not None:
        im3 = ax3.imshow(avg_stft_matrix3.T,
                         aspect='auto',
                         origin='lower',
                         extent=[stft_tarr[0]*1e3, stft_tarr[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
                         interpolation='None',
                         cmap='jet',
                         norm=colors.LogNorm(vmin=avg_stft_matrix3.T[avg_stft_matrix3.T > 0].min(), 
                                           vmax=avg_stft_matrix3.T.max()))
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Frequency (MHz)')
        ax3.text(0.98, 0.95, 'By_P20 (x=0cm, y=0cm)', transform=ax3.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
                 horizontalalignment='right', verticalalignment='top')
        fig.colorbar(im3, ax=ax3, label='Magnitude')

    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure as PNG
    ifn = r"C:\Users\hjia9\Documents\lapd\e-ring\diagnostic_fig"
    output_filename = os.path.join(ifn, "averaged_bdot_stft.png")
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved averaged Bdot STFT plot to: {output_filename}")
    
    plt.close(fig)  # Close the figure to free memory

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

    file_numbers = [f"{i:05d}" for i in range(0,11)] #+ [f"{i:05d}" for i in range(60, 89)]
    
    # base_dir = r"E:\good_data\kapton\He3kA_B380G800G_pl0t20_uw15t35"
    base_dir = r"E:\good_data\He3kA_B500G1kG_pl0t20_uw17t37_P30"

    # Uncomment one of these functions to run
    # main_plot(file_numbers, base_dir, debug=False)  # Process and display individual shots
    xray_wt_cam(file_numbers, base_dir, plot_Bdot=False, debug=False)
