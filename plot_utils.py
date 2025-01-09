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

def plot_stft(time_array, signal, samples_per_fft, overlap_fraction=0.5, window='hanning', 
             freq_min=0, freq_max=2000):
    """
    Compute and plot Short-Time Fourier Transform to show frequency evolution over time.
    Optimized version using vectorized operations and numpy's stride tricks.
    
    Args:
        time_array (np.ndarray): Time array in seconds
        signal (np.ndarray): Signal to analyze
        samples_per_fft (int): Number of samples to use in each FFT computation
        overlap_fraction (float): Fraction of overlap between segments (0 to 1)
        window (str): Window function to use ('hanning' or 'blackman')
        freq_min (float): Lower frequency limit in MHz (default 0 MHz)
        freq_max (float): Upper frequency limit in MHz (default 2000 MHz = 2 GHz)
    
    Returns:
        tuple: (segment_times, frequencies, STFT matrix)
    """
    # Calculate basic parameters
    dt = time_array[1] - time_array[0]  # Time step
    
    # Calculate overlap and hop size
    overlap = int(samples_per_fft * overlap_fraction)
    hop = samples_per_fft - overlap
    
    # Calculate resolutions
    time_resolution = dt * hop  # Time between successive FFTs
    freq_resolution = 1.0 / (dt * samples_per_fft) / 1e6  # Frequency resolution in MHz
    
    # Create window function
    if window.lower() == 'hanning':
        win = np.hanning(samples_per_fft)
    elif window.lower() == 'blackman':
        win = np.blackman(samples_per_fft)
    else:
        win = np.ones(samples_per_fft)
    
    # Pad signal if necessary
    pad_length = (samples_per_fft - len(signal)) % hop
    if pad_length > 0:
        signal = np.pad(signal, (0, pad_length), mode='constant')
    
    # Create strided array of segments using numpy's stride tricks
    shape = (samples_per_fft, (len(signal) - samples_per_fft) // hop + 1)
    strides = (signal.strides[0], signal.strides[0] * hop)
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    
    # Apply window to all segments at once
    segments = segments.T * win
    
    # Compute FFT for all segments at once
    stft_matrix = np.fft.rfft(segments, axis=1)
    
    # Compute magnitude (normalized)
    stft_matrix = 2.0/samples_per_fft * np.abs(stft_matrix)
    
    # Create frequency array in MHz
    freq = np.fft.rfftfreq(samples_per_fft, dt) / 1e6
    
    # Apply frequency mask for the specified range
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    freq = freq[freq_mask]
    stft_matrix = stft_matrix[:, freq_mask]
    
    # Create time array for segments (use center of each segment)
    segment_times = time_array[samples_per_fft//2:samples_per_fft//2 + stft_matrix.shape[0]*hop:hop]
    
    # Convert time to milliseconds
    segment_times_ms = segment_times * 1e3
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot spectrogram using imshow with time on x-axis
    im = plt.imshow(stft_matrix.T, 
                   aspect='auto',
                   origin='lower',
                   extent=[segment_times_ms[0], segment_times_ms[-1], freq[0], freq[-1]],
                   interpolation='nearest',
                   cmap='jet')
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Magnitude')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (MHz)')
    
    # Add resolution information to title
    title = f'Time-Frequency Analysis ({freq_min}-{freq_max} MHz)\nTime Resolution: {time_resolution*1e6:.1f} μs, Frequency Resolution: {freq_resolution:.2f} MHz'
    plt.title(title)

    
    return segment_times, freq, stft_matrix

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

def plot_counts_per_bin(pulse_times, bin_width_ms=5.0, figsize=(10,6)):
    """
    Plot total counts in each time bin.
    
    Args:
        pulse_times (np.ndarray): Array of pulse arrival times in milliseconds
        bin_width_ms (float): Width of time bins in milliseconds
        figsize (tuple): Figure size (width, height) in inches
    
    Returns:
        tuple: (bin_centers, counts) arrays where counts shows total counts in each bin
    """
    # Create time bins
    bins = np.arange(min(pulse_times), max(pulse_times) + bin_width_ms, bin_width_ms)
    bin_centers = (bins[:-1] + bins[1:])/2
    
    # Calculate histogram
    counts, _ = np.histogram(pulse_times, bins=bins)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot counts per bin as a bar plot
    plt.bar(bin_centers, counts, width=bin_width_ms*0.9, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Counts per Bin')
    plt.title(f'Photon Counts per {bin_width_ms} ms')
    plt.grid(True)
    
    # Add count rate information
    total_time = max(pulse_times) - min(pulse_times)
    count_rate = len(pulse_times) / (total_time/1000)  # Convert ms to s for rate
    plt.text(0.02, 0.98, 
             f'Average Count Rate: {count_rate:.1f} counts/s\n' + 
             f'Total Counts: {len(pulse_times)}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top')

    
    return bin_centers, counts

#==============================================================================
# Main Example
#==============================================================================

if __name__ == "__main__":
    data_path = r"E:\x-ray\20241029\C3--E-ring-wt-Tungsten2mm-xray--{shot:05d}.trc"
    figs = plot_shots_grid(data_path_template=data_path, shot_range=range(0, 5))

