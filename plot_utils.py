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
import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")

import tkinter as tk
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from typing import Callable, Optional
from read_scope_data import read_trc_data
import gc  # For garbage collection
import numpy as np


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

def position_window(window_manager, x_pos: int, y_pos: int, 
                   window_width: Optional[int] = None, 
                   window_height: Optional[int] = None) -> None:
    """
    Position a matplotlib window across different backends.
    
    Args:
        window_manager: plt.get_current_fig_manager() instance
        x_pos (int): Window x position
        y_pos (int): Window y position
        window_width (Optional[int]): Window width. If None, keeps current width
        window_height (Optional[int]): Window height. If None, keeps current height
    """
    try:
        # For Qt backend
        if window_width and window_height:
            window_manager.window.setGeometry(x_pos, y_pos, window_width, window_height)
        else:
            window_manager.window.move(x_pos, y_pos)
    except:
        try:
            # For TkAgg backend
            window_manager.window.wm_geometry(f"+{x_pos}+{y_pos}")
        except:
            try:
                # For WX backend
                window_manager.window.SetPosition((x_pos, y_pos))
            except:
                print("Could not position window - unsupported backend")

def select_monitor(monitor_idx: Optional[int] = None, 
                  window_scale: tuple = (1.0, 1.0),
                  position_fig: bool = True) -> tuple:
    """
    Select a monitor and optionally position the current matplotlib figure.
    
    Args:
        monitor_idx (Optional[int]): Monitor index to use. If None, will prompt user.
        window_scale (tuple): Scale factors (width, height) for window size.
        position_fig (bool): If True, positions the current matplotlib figure.
    
    Returns:
        tuple: (monitor_object, x_pos, y_pos, window_width, window_height)
    """
    monitors = get_monitors()
    
    # If monitor_idx not provided, show available monitors and prompt user
    if monitor_idx is None:
        print("\nAvailable monitors:")
        for i, m in enumerate(monitors):
            print(f"Monitor {i}: {m.width}x{m.height} at position ({m.x}, {m.y})")
        monitor_idx = int(input("\nEnter the monitor number to display plots on: "))
    
    # Validate monitor index
    if not 0 <= monitor_idx < len(monitors):
        raise ValueError(f"Invalid monitor index {monitor_idx}. Must be between 0 and {len(monitors)-1}")
    
    monitor = monitors[monitor_idx]
    
    # Calculate window dimensions
    width_scale, height_scale = window_scale
    window_width = int(monitor.width * width_scale)
    window_height = int(monitor.height * height_scale)
    
    # Calculate centered position
    x_pos = monitor.x + (monitor.width - window_width) // 2
    y_pos = monitor.y + (monitor.height - window_height) // 2
    
    # Position the current figure if requested
    if position_fig:
        mngr = plt.get_current_fig_manager()
        position_window(mngr, x_pos, y_pos, window_width, window_height)
    
    return monitor, x_pos, y_pos, window_width, window_height

#==============================================================================
# Display multiple shots on the screen
#==============================================================================
def plot_shots_grid(data_path_template: str, shot_range: range, n_cols: int = 2,
                   data_processor: Optional[Callable] = None, monitor_idx: Optional[int] = None) -> None:
    """Plot multiple shots in windows with 2x2 subplots."""
    # Clean up any existing figures first
    cleanup_figures()
    
    # Get monitor information using full window size
    monitor, base_x, base_y, window_width, window_height = select_monitor(
        monitor_idx=monitor_idx,
        window_scale=(0.9, 0.9)  # Use 90% of monitor size for each window
    )
    
    # Calculate number of windows needed (each window shows 4 plots)
    n_shots = len(shot_range)
    n_windows = (n_shots + 3) // 4  # Ceiling division by 4
    n_rows = (n_windows + n_cols - 1) // n_cols  # Ceiling division for window layout
    
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

def plot_counts_per_bin(bin_centers, counts, bin_width_ms, ax=None):
    """
    Plot number of pulses in each time bin.
    
    Args:
        bin_centers (np.ndarray): Array of bin center times in milliseconds
        counts (np.ndarray): Array of pulse counts per bin
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used.
    """
    if ax is None:
        ax = plt.gca()
        
    # Plot counts per bin
    ax.plot(bin_centers, counts, label=f'Counts per {bin_width_ms} ms')
    
    # Add labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Counts per Bin')
    ax.legend(loc='upper right')
    ax.grid(True)
    

#==============================================================================
# Photon Detection Functions
#==============================================================================

def plot_photon_detection(tarr, data, pulse_times, detector, ax=None):
    """
    Plot photon pulses from x-ray detector data with dynamic baseline.
    
    Args:
        time_array (np.ndarray): Time array in seconds
        signal_data (np.ndarray): Signal data to analyze
        pulse_times (np.ndarray): Array of detected pulse times in milliseconds
        detector (Photons): Photon detector object containing baseline and threshold
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Plot original signal
    ax.plot(tarr, data, 'b-', label='Original', alpha=0.7)
    
    # Plot dynamic baseline
    ax.plot(tarr, detector.baseline, 'g-', label='Baseline', alpha=0.7)
    
    # Plot baseline-subtracted signal
    subtracted_signal = data - detector.baseline
    ax.plot(tarr, subtracted_signal + detector.baseline.mean(), 'k-', 
            label='Subtracted', alpha=0.5)
    
    # Plot detected pulses on the subtracted signal
    pulse_heights = subtracted_signal[np.searchsorted(tarr, pulse_times)] + detector.baseline.mean()
    ax.plot(pulse_times, pulse_heights, 'r.', label='Pulses')
    
    # Plot threshold level
    ax.axhline(y=detector.baseline.mean() + detector.threshold, color='r', 
               linestyle='--', alpha=0.5, label='Threshold')
    
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'Detected {len(pulse_times)} pulses')
    ax.legend(loc='upper right')
    ax.grid(True)

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

def plot_original_and_baseline(tarr, data, detector, ax=None):
    """Plot original signal and its calculated baseline.
    
    Args:
        tarr: Time array in milliseconds
        data: Original signal data
        detector: Photons detector object containing baseline
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        ax = plt.gca()
    
    # Plot original signal and baseline
    ax.plot(tarr, data, 'b-', alpha=0.7, label='Original')
    ax.plot(tarr, detector.baseline, 'r-', alpha=0.7, label='Baseline')
    
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_subtracted_signal(tarr, data, pulse_times, detector, ax=None):
    """Plot baseline-subtracted signal with detected pulses.
    
    Args:
        tarr: Time array in milliseconds
        data: Original signal data (not used, kept for API consistency)
        pulse_times: Not used, kept for API consistency
        detector: Photons detector object containing baseline-subtracted signal
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        ax = plt.gca()
    
    # Plot baseline-subtracted signal
    ax.plot(tarr, detector.baseline_subtracted, 'b-', alpha=0.7, label='Baseline Subtracted')
    
    # Plot detected pulses at their actual times and heights
    if hasattr(detector, 'pulses'):
        # Plot reduced pulses
        times = np.array([p.time for p in detector.pulses])
        areas = np.array([p.area for p in detector.pulses])
        ax.plot(times, areas, 'r.', markersize=2, label='Detected Pulses')
    
    # Plot threshold level
    ax.axhline(y=detector.threshold, color='g', linestyle='--', 
               alpha=0.5, label='Threshold')

    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_stft_wt_photon_counts(tarr, fft_arr, freq_arr, bin_centers, counts, fig=None, ax=None):
    """Plot STFT spectrogram with photon counts overlay.
    
    Args:
        tarr: Time array for STFT in seconds
        fft_arr: 2D array of STFT values
        freq_arr: Frequency array in Hz
        bin_centers: Time array for photon counts in milliseconds
        counts: Array of photon counts
        fig: Figure object for colorbar
        ax: Axis to plot on
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    # Plot STFT
    im = ax.imshow(fft_arr.T, 
                   aspect='auto',
                   origin='lower',
                   extent=[tarr[0]*1e3, tarr[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
                   interpolation='None',
                   cmap='jet')
    
    # Add colorbar
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    cax = fig.add_axes([pos.x0 + pos.width * 0.92, pos.y0, 0.02, pos.height])
    fig.colorbar(im, cax=cax)
    
    # Add counts overlay
    ax_twin = ax.twinx()
    ax_twin.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax_twin.plot(bin_centers, counts, 'w-', linewidth=1, alpha=0.7)
    ax_twin.set_yticks([])
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('STFT with Photon Counts Overlay')

    ax.set_xlim(tarr[0]*1e3, tarr[-1]*1e3)

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":
    data_path = r"E:\x-ray\20241029\C3--E-ring-wt-Tungsten2mm-xray--{shot:05d}.trc"
    figs = plot_shots_grid(data_path_template=data_path, shot_range=range(0, 5))

