'''
Plotting Utilities Module
------------------------

This module provides various plotting functions for data analysis. Functions are organized
into the following categories:

1. Frequency Analysis
    - plot_fft: Plot Fast Fourier Transform of signals
    - plot_stft: Plot Short-Time Fourier Transform (spectrogram)

2. Multi-Shot Display
    - plot_shots_grid: Display multiple shots in a grid layout across monitors
    - cleanup_figures: Clean up matplotlib figures and free memory

3. Photon Counting
    - plot_counts_per_bin: Plot histogram of photon counts in time bins

Each function includes detailed documentation of its parameters and returns.
'''

import tkinter as tk
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from typing import Callable, Optional
from read_scope_data import read_trc_data
import gc  # For garbage collection
import numpy as np

#==============================================================================
# Frequency Analysis Functions
#==============================================================================

def plot_fft(time_array, signals_dict, window=None):
    """
    Compute and plot FFT of signals with proper frequency units.
    
    Args:
        time_array (np.ndarray): Time array in seconds
        signals_dict (dict): Dictionary of signals to plot with their labels
        window (str, optional): Window function to use (e.g., 'hanning', 'blackman')
    
    Returns:
        tuple: (frequencies in MHz, magnitude spectrum)
    """
    # Calculate sampling parameters
    dt = time_array[1] - time_array[0]  # Time step
    fs = 1/dt  # Sampling frequency
    n = len(time_array)
    
    # Create frequency array in MHz
    freq = np.fft.rfftfreq(n, dt) / 1e6  # Convert to MHz
    
    plt.figure(figsize=(10, 6))
    
    for label, signal in signals_dict.items():
        # Apply window if specified
        if window is not None:
            if window.lower() == 'hanning':
                win = np.hanning(len(signal))
            elif window.lower() == 'blackman':
                win = np.blackman(len(signal))
            signal = signal * win
            
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        # Compute magnitude spectrum (normalized)
        magnitude = 2.0/n * np.abs(fft_result)
        
        # Plot
        plt.plot(freq, magnitude, label=label)
    
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend(loc='upper right')  # Fixed legend location
    
    return freq, magnitude


#==============================================================================
# Multi-Shot Display Functions
#==============================================================================

def cleanup_figures():
    """
    Helper function to properly clean up matplotlib figures and references.
    Useful for freeing memory in Jupyter notebooks.
    """
    # Close all figures
    plt.close('all')
    # Clear the current figure
    plt.clf()
    # Clear the current axes
    plt.cla()
    # Force garbage collection
    gc.collect()

def plot_shots_grid(
    data_path_template: str,
    shot_range: range,
    n_cols: int = 2,
    data_processor: Optional[Callable] = None,
    monitor_idx: Optional[int] = None
) -> None:
    """
    Plot multiple shots in windows with 2x2 subplots, spread across monitors.
    Memory efficient version that clears data after each plot.
    
    Args:
        data_path_template (str): Template string for data file paths with {shot:05d} format
        shot_range (range): Range of shot numbers to plot
        n_cols (int): Number of windows to arrange horizontally
        data_processor (Callable, optional): Function to process data before plotting
        monitor_idx (Optional[int]): Monitor index to display plots on. If None, will prompt user
    
    Returns:
        list: List of figure objects for cleanup
    """
    # Clean up any existing figures first
    cleanup_figures()
    
    # Get monitor information
    monitors = get_monitors()
    
    # If monitor_idx not provided, show available monitors and prompt user
    if monitor_idx is None:
        print("\nAvailable monitors:")
        for i, m in enumerate(monitors):
            print(f"Monitor {i}: {m.width}x{m.height} at position ({m.x}, {m.y})")
        monitor_idx = int(input("\nEnter the monitor number to display plots on: "))
    
    target_monitor = monitors[monitor_idx]
    
    # Calculate number of windows needed (each window shows 4 plots)
    n_shots = len(shot_range)
    n_windows = (n_shots + 3) // 4  # Ceiling division by 4
    n_rows = (n_windows + n_cols - 1) // n_cols  # Ceiling division for window layout
    
    # Calculate window sizes
    window_width = target_monitor.width // n_cols
    window_height = target_monitor.height // n_rows
    
    # Base position for windows
    base_x = target_monitor.x
    base_y = target_monitor.y

    # Find common path prefix for naming windows
    first_path = data_path_template.format(shot=shot_range[0])
    path_parts = first_path.split('/')
    common_prefix = '/'.join(path_parts[:-1]) + '/'
    
    figures = []  # Keep track of figures for cleanup
    
    # Create windows with 2x2 subplots
    for window_idx in range(n_windows):
        # Create new figure with 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        figures.append(fig)  # Add to list for cleanup
        axs = axs.ravel()  # Flatten axes array for easier indexing
        
        # Calculate window position
        row = window_idx // n_cols
        col = window_idx % n_cols
        x_pos = base_x + (col * window_width)
        y_pos = base_y + (row * window_height)
        
        # Plot 4 shots in this window
        for subplot_idx in range(4):
            shot_idx = window_idx * 4 + subplot_idx
            if shot_idx < n_shots:
                shot = shot_range[shot_idx]
                
                # Read data
                data_path = data_path_template.format(shot=shot)
                data, tarr = read_trc_data(data_path, False)
                
                # Process data if processor provided
                if data_processor:
                    data = data_processor(data)
                
                # Plot data
                axs[subplot_idx].plot(tarr, -data)
                axs[subplot_idx].set_title(f'Shot {shot}')
                axs[subplot_idx].set_xlabel('Time')
                axs[subplot_idx].set_ylabel('Signal')
                
                # Clear data from memory
                del data
                del tarr
                gc.collect()  # Force garbage collection
            else:
                # Hide empty subplots
                axs[subplot_idx].set_visible(False)
        
        # Position window
        mngr = plt.get_current_fig_manager()
        # Set window title using common path
        try:
            mngr.set_window_title(common_prefix)
        except:
            print(f"Could not set window title for {common_prefix}")
            
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
        
        plt.show(block=False)
        
        # Force garbage collection of data (but keep the figure)
        gc.collect()
    
    # Create a small tkinter window to keep the plots alive without blocking
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter window
    root.quit()  # Allow the script to terminate while keeping plots open
    
    # Return the list of figures so they can be properly cleaned up later
    return figures

#==============================================================================
# Photon Counting Functions
#==============================================================================

def plot_counts_per_bin(pulse_times, pulse_areas, bin_width_ms=5.0, 
                    amplitude_min=None, amplitude_max=None, ax=None):
    """
    Plot total counts in each time bin with optional amplitude threshold filtering.
    
    Args:
        pulse_times (np.ndarray): Array of pulse arrival times in milliseconds
        pulse_areas (np.ndarray): Array of pulse areas/amplitudes
        bin_width_ms (float): Width of time bins in milliseconds
        amplitude_min (float, optional): Minimum amplitude threshold for counting pulses
        amplitude_max (float, optional): Maximum amplitude threshold for counting pulses
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used.
    
    Returns:
        tuple: (bin_centers, counts) arrays where counts shows total counts in each bin
    """
    if ax is None:
        ax = plt.gca()
        
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
        pulse_areas = pulse_areas[mask]
    
    # Create time bins
    bins = np.arange(min(pulse_times), max(pulse_times) + bin_width_ms, bin_width_ms)
    bin_centers = (bins[:-1] + bins[1:])/2
    
    # Calculate histogram
    counts, _ = np.histogram(pulse_times, bins=bins)
    
    # Plot counts per bin
    ax.plot(bin_centers, counts)
    
    # Add labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Counts per Bin')
    title = f'Photon Counts per {bin_width_ms} ms'
    if amplitude_min is not None or amplitude_max is not None:
        threshold_text = ''
        if amplitude_min is not None:
            threshold_text += f'Min: {amplitude_min:.3f}'
        if amplitude_max is not None:
            if threshold_text:
                threshold_text += ', '
            threshold_text += f'Max: {amplitude_max:.3f}'
        title += f'\nAmplitude Thresholds: {threshold_text}'
    ax.set_title(title)
    ax.grid(True)
    
    # Add count rate and signal level information
    total_time = max(pulse_times) - min(pulse_times)
    count_rate = len(pulse_times) / (total_time/1000)  # Convert ms to s for rate
    info_text = (f'Average Count Rate: {count_rate:.1f} counts/s\n'
                f'Total Counts: {len(pulse_times)}\n'
                f'Min Signal: {min(pulse_areas):.3f}\n'
                f'Max Signal: {max(pulse_areas):.3f}')
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    return bin_centers, counts

#==============================================================================
# Photon Detection Functions
#==============================================================================

def plot_photon_detection(tarr, data, pulse_times, detector, ax=None):
    """
    Plot photon pulses from x-ray detector data.
    
    Args:
        time_array (np.ndarray): Time array in seconds
        signal_data (np.ndarray): Signal data to analyze
        pulse_times (np.ndarray): Array of detected pulse times in milliseconds
        threshold_level (float): Detection threshold level
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.plot(tarr, data, 'b-', label='Signal')
    ax.plot(pulse_times, [detector.threshold + detector.offset]*len(pulse_times), 
            'r.', label='Detected Pulses')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Signal')
    ax.set_title(f'Detected {len(pulse_times)} pulses')
    ax.legend(loc='upper right')
    ax.grid(True)
    

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":
    data_path = r"E:\x-ray\20241029\C3--E-ring-wt-Tungsten2mm-xray--{shot:05d}.trc"
    figs = plot_shots_grid(data_path_template=data_path, shot_range=range(0, 5))

