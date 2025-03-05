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
    fig = plt.figure(fig_id)
    
    # Position the window on the specified monitor
    # select_monitor(monitor_idx=monitor_idx, window_scale=(0.1, 0.1))
    
    # Create 2 subplots
    gs = GridSpec(2, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
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
        ax1.plot(tarr_I*1e3, P_data + Pref_data*1000, 'b-', label='Power')
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Power (W)', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.grid(True)
        ax1.set_title('Power Plot', fontsize=14, pad=10)
        
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
    
    # Calculate photon counts per bin
    bin_centers, counts = detector.counts_per_bin(bin_width_ms=0.1)
    
    # Free memory
    del xray_data
    del tarr_x
    
    # Plot 2: STFT from Bdot data
    # First try to load specified channel, then Channel 2 as fallback
    bdot_data = None
    tarr_B = None
    channel_used = None
    
    # Try to load the requested channel first
    print(f"Attempting to load Bdot data from Channel {bdot_channel}")
    for f in bdot_files:
        if f"C{bdot_channel}--" in f:
            filepath = os.path.join(base_dir, f)
            try:
                bdot_data, tarr_B = read_trc_data(filepath)
                channel_used = f"C{bdot_channel}"
                print(f"Using Bdot data from Channel {bdot_channel}: {f}")
                break
            except Exception as e:
                print(f"Error loading Bdot data from Channel {bdot_channel}: {e}")
                
    # If requested channel failed, try channel 2 as fallback
    if bdot_data is None or tarr_B is None:
        print("Falling back to Channel 2 for Bdot data")
        for f in bdot_files:
            if "C2--" in f:
                filepath = os.path.join(base_dir, f)
                try:
                    bdot_data, tarr_B = read_trc_data(filepath)
                    channel_used = "C2"
                    print(f"Using Bdot data from Channel 2 as fallback: {f}")
                    break
                except Exception as e:
                    print(f"Error loading Bdot data from Channel 2: {e}")
    
    # Calculate and plot STFT if Bdot data is available
    if bdot_data is not None and tarr_B is not None:
        try:
            # Calculate STFT using parameters from xray_test.ipynb
            freq, stft_matrix, stft_time = calculate_stft(
                time_array=tarr_B, 
                data_arr=bdot_data,
                freq_bins=500, 
                overlap_fraction=0.05,
                window='hanning', 
                freq_min=100e6, 
                freq_max=800e6
            )
            
            # Plot STFT with photon counts overlay
            plot_stft_wt_photon_counts(stft_time, stft_matrix, freq, bin_centers, counts, fig=fig, ax=ax2)
            
            # Adjust STFT plot labels for better visibility
            ax2.set_xlim(10,14)
            
        except Exception as e:
            print(f"Error calculating STFT: {e}")
            ax2.text(0.5, 0.5, f'Error calculating STFT: {str(e)}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'Bdot data not available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('STFT Plot (Data Not Available)', fontsize=14, pad=10)
    
    # Add some padding around the entire figure
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    
    # Draw the figure to ensure it shows up
    fig.canvas.draw()
    plt.pause(0.1)
    
    # Reset the interactive mode to what it was
    if not old_interactive:
        plt.ioff()
    
    return fig

def process_shot_2(file_number, base_dir, bdot_channel=1):
    """Process a single shot and return data for averaging."""
    
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
    
    # Initialize variables
    P_data = None
    tarr = None

    I_data = None
    for f in power_files:
        if f"C2--" in f:
            filepath = os.path.join(base_dir, f)
            I_data, tarr = read_trc_data(filepath)
            print(f"Using current file: {f}")
            break
    
    # Find voltage data (Channel 4)
    V_data = None
    for f in power_files:
        if f"C4--" in f:
            filepath = os.path.join(base_dir, f)
            V_data, tarr = read_trc_data(filepath)
            print(f"Using voltage file: {f}")
            break
    
    # Find reflected power data (Channel 3)
    Pref_data = None
    for f in power_files:
        if f"C3--" in f:
            filepath = os.path.join(base_dir, f)
            Pref_data, _ = read_trc_data(filepath)  # Use underscore to ignore time array
            print(f"Using reflected power file: {f}")
            break
    
    P_data = (I_data * 0.25) * (-V_data)  # 1V ~ 0.25A; V is recorded negative
    P_data = P_data + Pref_data*1000 # Total - reflected power
 
    # Find X-ray data
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
    bin_centers, counts= detector.counts_per_bin(0.1)
    
    # Bdot data
    bdot_data = None
    tarr_B = None
    freq = None
    stft_matrix = None
    stft_time = None
    
    print(f"Attempting to load Bdot data from Channel {bdot_channel}")
    for f in bdot_files:
        if f"C{bdot_channel}--" in f:
            filepath = os.path.join(base_dir, f)
            bdot_data, tarr_B = read_trc_data(filepath)
            print(f"Using Bdot data from Channel {bdot_channel}: {f}")

    
    # Calculate and plot STFT if Bdot data is available
    if bdot_data is not None and tarr_B is not None:
            # Calculate STFT using parameters from xray_test.ipynb
        freq, stft_matrix, stft_time = calculate_stft(
            time_array=tarr_B, 
            data_arr=bdot_data,
            freq_bins=500, 
            overlap_fraction=0.05,
            window='hanning', 
            freq_min=100e6, 
            freq_max=800e6
        )
    
    return tarr, P_data, bin_centers, counts, freq, stft_matrix, stft_time
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

def main_average(file_number, base_dir):
    
    print(f"Starting processing for {len(file_numbers)} shots")
    print(f"Data directory: {base_dir}")

    # Initialize arrays to store data for averaging
    all_P_data = []
    all_counts = []
    all_stft_matrices = []
    
    # Common time and frequency arrays
    common_tarr = None
    common_bin_centers = None
    common_freq = None
    common_stft_time = None
    
    processed_shots = 0
    
    for file_number in file_numbers:
        try:
            tarr, P_data, bin_centers, counts, freq, stft_matrix, stft_time = process_shot_2(file_number, base_dir, bdot_channel=2)
            
            # Check if all required data is available
            if bin_centers is None or counts is None:
                print(f"Missing count data for shot {file_number}, skipping")
                continue
                
            if freq is None or stft_matrix is None or stft_time is None:
                print(f"Missing STFT data for shot {file_number}, skipping")
                continue
                
            # Store the first valid shot's time and frequency arrays as reference
            if common_tarr is None and tarr is not None:
                common_tarr = tarr
            
            if common_bin_centers is None:
                common_bin_centers = bin_centers
                
            if common_freq is None:
                common_freq = freq
                
            if common_stft_time is None:
                common_stft_time = stft_time
            
            # Append data for averaging (only if they exist)
            if P_data is not None and tarr is not None:
                all_P_data.append(P_data)
            
            all_counts.append(counts)
            all_stft_matrices.append(stft_matrix)
            
            processed_shots += 1
            print(f"Successfully processed shot {file_number} ({processed_shots}/{len(file_numbers)})")
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error processing shot {file_number}: {e}")
        except KeyboardInterrupt:
            print("Process terminated by KeyboardInterrupt")
            break

    # Check if we have enough data to proceed
    if processed_shots == 0:
        print("No shots were successfully processed. Exiting.")
        return None, None
        
    print(f"Successfully processed {processed_shots} shots")
    
    # Calculate averages
    avg_P_data = None
    if len(all_P_data) > 0:
        try:
            avg_P_data = np.mean(all_P_data, axis=0)
            print(f"Averaged power data from {len(all_P_data)} shots")
        except Exception as e:
            print(f"Error averaging power data: {e}")
    else:
        print("No power data available for averaging")
    
    avg_counts = None
    std_counts = None
    if len(all_counts) > 0:
        try:
            # Make sure all count arrays are the same length by using the shortest one
            min_count_length = min(len(counts) for counts in all_counts)
            print(f"Trimming count arrays to length {min_count_length}")
            
            trimmed_counts = [counts[:min_count_length] for counts in all_counts]
            common_bin_centers = common_bin_centers[:min_count_length]
            
            avg_counts = np.sum(trimmed_counts, axis=0)
            std_counts = np.std(trimmed_counts, axis=0)
            print(f"Averaged count data from {len(all_counts)} shots")
        except Exception as e:
            print(f"Error averaging count data: {e}")
    else:
        print("No count data available for averaging")
        
    avg_stft_matrix = None
    if len(all_stft_matrices) > 0:
        try:
            # Verify all matrices have the same shape
            first_shape = all_stft_matrices[0].shape
            valid_matrices = [m for m in all_stft_matrices if m.shape == first_shape]
            
            if len(valid_matrices) != len(all_stft_matrices):
                print(f"Warning: {len(all_stft_matrices) - len(valid_matrices)} STFT matrices had inconsistent shapes and were excluded")
                
            if len(valid_matrices) > 0:
                # Convert to numpy array for easier averaging
                stft_array = np.array(valid_matrices)
                avg_stft_matrix = np.mean(stft_array, axis=0)
                print(f"Averaged STFT data from {len(valid_matrices)} shots")
            else:
                print("No valid STFT matrices for averaging")
        except Exception as e:
            print(f"Error averaging STFT data: {e}")
    else:
        print("No STFT data available for averaging")
    
    # Check if we have data to plot
    if avg_stft_matrix is None and avg_counts is None:
        print("No averaged data available for plotting. Exiting.")
        return None, None
    
    # Create figures for average results
    plt.ion()  # Turn on interactive mode
    
    fig1 = None
    fig2 = None
    
    # Figure 1: Average STFT matrix (if available)
    if avg_stft_matrix is not None and common_stft_time is not None and common_freq is not None:
        fig1 = plt.figure(figsize=(14, 10))
        ax1 = fig1.add_subplot(111)
        
        # Plot STFT
        im = ax1.imshow(avg_stft_matrix.T, 
                       aspect='auto',
                       origin='lower',
                       extent=[common_stft_time[0]*1e3, common_stft_time[-1]*1e3, 
                              common_freq[0]/1e6, common_freq[-1]/1e6],
                       interpolation='None',
                       cmap='jet')
        
        # Add colorbar
        cbar = fig1.colorbar(im, ax=ax1)
        cbar.set_label('Average Magnitude')
        
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Frequency (MHz)', fontsize=12)
        ax1.set_title(f'Average STFT Spectrogram ({len(all_stft_matrices)} shots)', fontsize=14)
        
        # Set useful time range
        ax1.set_xlim(10, 14)
        
        # Draw the figure to ensure it shows up
        fig1.canvas.draw()
    else:
        print("Cannot create STFT figure - missing data")
    
    # Figure 2: Photon Counts total
    if avg_counts is not None and std_counts is not None and common_bin_centers is not None:
        fig2 = plt.figure(figsize=(14, 10))
        ax2 = fig2.add_subplot(111)
        
        # Plot average counts
        ax2.plot(common_bin_centers, avg_counts, 'b-', linewidth=2)
        # Add error bars (standard deviation)
        # ax2.fill_between(common_bin_centers, avg_counts - std_counts, avg_counts + std_counts, 
        #                 alpha=0.3, color='blue', label='±1 Std Dev')
        
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Counts per Bin', fontsize=12)
        ax2.set_title(f'Photon Counts sum({len(all_counts)} shots)', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        # Draw the figure to ensure it shows up
        fig2.canvas.draw()
    else:
        print("Cannot create Counts figure - missing data")
    
    # Make sure at least one figure was created
    if fig1 is None and fig2 is None:
        print("No figures were created. Exiting.")
        return None, None
    
    # Create a small tkinter window to keep the plots alive without blocking
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    # Add explicit pause to ensure the display updates
    plt.pause(0.5)
    
    # Turn off interactive mode and show all figures
    plt.ioff()
    plt.show(block=True)  # Block until all figures are closed
    
    return fig1, fig2

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

    file_numbers = [f"{i:05d}" for i in range(11,27)]
    # base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t45_P24"
    base_dir = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30"

    # Uncomment one of these functions to run
    # main_plot(file_numbers, base_dir, debug=False)  # Process and display individual shots
    main_average(file_numbers, base_dir)  # Process, average and display aggregate data
