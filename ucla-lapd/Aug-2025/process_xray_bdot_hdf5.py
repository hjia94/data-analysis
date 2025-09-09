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
from unittest import result
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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__ if '__file__' in globals() else os.getcwd()), '../..'))
sys.path = [repo_root, f"{repo_root}/read", f"{repo_root}/object_tracking"] + sys.path

from read.read_scope_data import read_trc_data, read_hdf5_all_scopes_channels, read_scope_channel_descriptions
from data_analysis_utils import Photons, calculate_stft, counts_per_bin
from object_tracking.read_cine import read_cine, convert_cine_to_avi
from object_tracking.track_object import track_object, detect_chamber, get_vel_freefall, get_pos_freefall

#===========================================================================================================
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

#===========================================================================================================

def get_magnetron_power_data(f, result, scope_name='magscope'):
    """
    Calculate magnetron power from HDF5 file data.
    """
    if scope_name not in result:
        print(f"Scope '{scope_name}' not found.")
        return None, None, None, None, None

    tarr = result[scope_name].get('time_array')
    chan_data = result[scope_name].get('channels', {})
    descriptions = read_scope_channel_descriptions(f,'magscope')

    I_data = None
    V_data = None
    Pref_data = None

    for ch, desc in descriptions.items():
        if 'current' in desc:
            I_data = chan_data[ch]
            if isinstance(desc, str):
                m = re.search(r'(\d+\.?\d*)\s*a/v', desc.lower())
                if m:
                    scale_factor = float(m.group(1))
                    print(f"Applying current scaling factor: {scale_factor} A/V")
                    I_data = I_data * scale_factor

        if 'voltage' in desc:
            V_data = chan_data[ch]
        if 'pref' in desc:
            Pref_data = chan_data[ch]

    P_data = None
    if I_data is not None and V_data is not None:
        P_data = ndimage.gaussian_filter1d(I_data * (-V_data) * 0.6, sigma=100)
        print(f"Magnetron power calculated")
    else:
        print("Cannot calculate power: missing current or voltage data")

    return tarr, P_data

def get_xray_data(result, scope_name = 'xrayscope'):
    tarr_x = result[scope_name].get('time_array')
    xray_data = result[scope_name]['channels']['C2']
    return tarr_x, xray_data

def get_bdot_data(result, scope_name='bdotscope'):

    tarr = result[scope_name].get('time_array')
    chan_data = result[scope_name].get('channels', {})
    descriptions = read_scope_channel_descriptions(f, scope_name)

    # for ch, desc in descriptions.items():

    #     m = re.search(r'P(\d+)', desc)
    #     if m:
    #         p_number = int(m.group(1))

    return tarr, chan_data

def calculate_bdot_stft(tarr_B, bdot_data, channel_info, freq_bins=1000, overlap_fraction=0.05, freq_min=200e6, freq_max=2000e6):
    '''
    Calculates STFT for each Bdot signal in bdot_data.
    Returns: stft_time, freq, stft_matrix1, stft_matrix2, stft_matrix3, stft_matrices
    '''
    stft_matrices = {}
    stft_time = None
    freq = None
    for channel, data in bdot_data.items():
        if data is not None:
            freq, stft_matrix, stft_time = calculate_stft(tarr_B, data, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
            stft_matrices[channel] = stft_matrix
        else:
            stft_matrices[channel] = None
    channels = sorted([ch for ch in channel_info.keys() if ch.startswith('C')])
    stft_matrix1 = stft_matrices.get(channels[0]) if len(channels) > 0 else None
    stft_matrix2 = stft_matrices.get(channels[1]) if len(channels) > 1 else None
    stft_matrix3 = stft_matrices.get(channels[2]) if len(channels) > 2 else None
    return stft_time, freq, stft_matrix1, stft_matrix2, stft_matrix3, stft_matrices

def process_shot_xray(tarr_x, xray_data, min_ts, d, threshold, debug=False):
    """Process a single shot of xray data and return data for averaging.
    """

    detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)
    detector.reduce_pulses()

    return detector.pulse_times, detector.pulse_amplitudes

def process_video(base_dir, cam_file):
    '''
    Process the video file for a given shot number and return the time at which the ball reaches the chamber center
    '''
    
    filepath = os.path.join(base_dir, cam_file)
    avi_path = os.path.join(base_dir, f"{os.path.splitext(cam_file)[0]}.avi")

    # Initialize or load the tracking dictionary
    tracking_dict = {}
    tracking_file = os.path.join(base_dir, f"tracking_{cam_file[:2]}.npy")

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
    return ct, frame, (cx, cy), chamber_radius

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

def xray_wt_cam(base_dir, fn, plot_Bdot=False, debug=False):
    # Define path
    ifn = os.path.join(base_dir, fn)
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
    
    with h5py.File(ifn, 'r') as f:
        shot_numbers = f['Control/FastCam']['shot number'][()]

    for shot_num in shot_numbers:
        print(f"\nProcessing shot {shot_num}")        

        with h5py.File(ifn, 'r') as f:
            print(f"Reading data from {ifn}")
            cine_narr = f['Control/FastCam/cine file name'][()]
            cam_file = cine_narr[shot_num-1]
            if shot_num != int(cam_file[-8:-5]):
                print(f"Warning: shot number {shot_num} does not match {cam_file}")

            result = read_hdf5_all_scopes_channels(f, shot_num, include_tarr=True)
            tarr_P, P_data = get_magnetron_power_data(f, result)
            tarr_x, xray_data = get_xray_data(result)

        try:
            t0, frame, (cx, cy), chamber_radius = process_video(base_dir, cam_file.decode('utf-8'))
        except FileNotFoundError as e:
            print(f"No video file found for shot {shot_num}; skipping...")
            continue

        # Create individual figure with two subplots (video + power)
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5), num=f"shot_{shot_num}")

        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        chamber_circle = plt.Circle((cx, cy), chamber_radius, fill=False, color='green', linewidth=2)
        ax1.add_patch(chamber_circle)
        print(f"ball reaches chamber center at t={t0 * 1e3:.3f}ms from plasma trigger")
        ax1.axis('off')

        # Plot power data if available
        if P_data is not None and tarr_P is not None:
            ax3.plot(tarr_P*1e3, P_data*1e-4, 'b-', linewidth=2)
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
        if shot_num in analysis_dict:
            pulse_tarr, pulse_amp = analysis_dict[shot_num]
        else:
            threshold = [5, 50]
            min_ts = 0.8e-6
            d = 0.1
            pulse_tarr, pulse_amp = process_shot_xray(tarr_x, xray_data, min_ts, d, threshold, debug=debug)

            # Save new results
            analysis_dict[shot_num] = (pulse_tarr, pulse_amp)
            try:
                np.save(analysis_file, analysis_dict)
                print(f"Added new analysis result for shot {shot_num}")
            except Exception as e:
                print(f"Error saving analysis results: {e}")
        
        # Store ALL pulse data for plot in the end
        shot_data = {
            'shot_num': shot_num,
            't0': t0,
            'uw_start': 30,
            'pulse_tarr': pulse_tarr,
            'pulse_amp': pulse_amp,
        }
        
        all_scatter_data.append(shot_data)

    
        # if plot_Bdot:
        #     try:
        #         stft_tarr, freq_arr, stft_matrix1, stft_matrix2, stft_matrix3 = process_shot_bdot(file_number, base_dir, debug=debug)
        #     except FileNotFoundError as e:
        #         print(f"No Bdot data found for shot {file_number}; skipping...")
        #         continue
            
        #     # Collect STFT matrices for averaging
        #     if stft_matrix1 is not None:
        #         all_stft_matrix1.append(stft_matrix1)
        #     if stft_matrix2 is not None:
        #         all_stft_matrix2.append(stft_matrix2)
        #     if stft_matrix3 is not None:
        #         all_stft_matrix3.append(stft_matrix3)

        #     # Average the STFT matrices with optional time axis limitation
        #     avg_stft_matrix1 = None
        #     avg_stft_matrix2 = None
        #     avg_stft_matrix3 = None
            
        #     # Option to limit time axis to first half (set to True to enable)
        #     limit_time_axis = False
            
        #     if len(all_stft_matrix1) > 0:
        #         matrices = np.array(all_stft_matrix1)
        #         if limit_time_axis:
        #             index = matrices.shape[1] // 2
        #             matrices = matrices[:, :index, :]
        #         avg_stft_matrix1 = np.mean(matrices, axis=0)
        #         print(f"Averaged {len(all_stft_matrix1)} By_P21 STFT matrices")
            
        #     if len(all_stft_matrix2) > 0:
        #         matrices = np.array(all_stft_matrix2)
        #         if limit_time_axis:
        #             index = matrices.shape[1] // 2
        #             matrices = matrices[:, :index, :]
        #         avg_stft_matrix2 = np.mean(matrices, axis=0)
        #         print(f"Averaged {len(all_stft_matrix2)} Bx_P20 STFT matrices")
            
        #     if len(all_stft_matrix3) > 0:
        #         matrices = np.array(all_stft_matrix3)
        #         if limit_time_axis:
        #             index = matrices.shape[1] // 2
        #             matrices = matrices[:, :index, :]
        #         avg_stft_matrix3 = np.mean(matrices, axis=0)
        #         print(f"Averaged {len(all_stft_matrix3)} By_P20 STFT matrices")
            
        #     if limit_time_axis:
        #         stft_tarr = stft_tarr[:index]
        #         freq_arr = freq_arr[:index]

        #     plot_averaged_bdot_stft(avg_stft_matrix1, avg_stft_matrix2, avg_stft_matrix3, stft_tarr, freq_arr)
    
    
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
            
        r_arr = get_pos_freefall(bin_centers + shot_data['uw_start'], shot_data['t0'])
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

def process_shot_bdot(file_number, base_dir, debug=False):
    """
    Process Bdot signals for a specific shot number.
    Returns the STFT time array, frequency array, and STFT matrices for three channels.
    
    Args:
        file_number: The shot number to process
        base_dir: Base directory containing the HDF5 files
        debug: Whether to print debug information
        
    Returns:
        stft_tarr: Time array for the STFT
        freq_arr: Frequency array for the STFT
        stft_matrix1, stft_matrix2, stft_matrix3: STFT matrices for three channels
    """
    # Construct the HDF5 filename
    hdf5_filename = os.path.join(base_dir, f'{file_number}.hdf5')
    
    if not os.path.exists(hdf5_filename):
        print(f"Error: HDF5 file not found: {hdf5_filename}")
        return None, None, None, None, None
    
    # Read the Bdot signals
    tarr_B, bdot_data, channel_info = read_bdot_signals_from_hdf5(hdf5_filename, debug=debug)
    
    if tarr_B is None or len(bdot_data) == 0:
        print(f"Error: Failed to read Bdot signals from {hdf5_filename}")
        return None, None, None, None, None
    
    # Calculate the STFT for each channel
    stft_tarr, freq_arr, stft_matrices = calculate_bdot_stft(tarr_B, bdot_data, channel_info)
    
    # Extract the STFT matrices for the three channels (assuming they exist)
    channels = list(stft_matrices.keys())
    if len(channels) >= 3:
        stft_matrix1 = stft_matrices[channels[0]]
        stft_matrix2 = stft_matrices[channels[1]]
        stft_matrix3 = stft_matrices[channels[2]]
    else:
        # Handle the case where we don't have three channels
        stft_matrix1 = stft_matrices.get(channels[0], None) if channels else None
        stft_matrix2 = stft_matrices.get(channels[1], None) if len(channels) > 1 else None
        stft_matrix3 = stft_matrices.get(channels[2], None) if len(channels) > 2 else None
    
    return stft_tarr, freq_arr, stft_matrix1, stft_matrix2, stft_matrix3

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

    base_dir = r"F:\AUG2025\P24"
    fn = "00_He1kG430G_5800A_K-25_2025-08-12.hdf5"

    xray_wt_cam(base_dir, fn, plot_Bdot=False, debug=False)