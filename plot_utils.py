'''
This script is used to for different plotting functions.
'''

import tkinter as tk
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from typing import Callable, Optional
from read_scope_data import read_trc_data
import gc  # For garbage collection

def cleanup_figures():
    """Helper function to properly clean up matplotlib figures and references."""
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
    n_cols: int = 2,  # Number of windows horizontally
    data_processor: Optional[Callable] = None,
    monitor_idx: Optional[int] = None
) -> None:
    """
    Plot multiple shots in windows with 2x2 subplots, spread across monitors.
    Memory efficient version that clears data after each plot.
    For Jupyter notebooks: Call cleanup_figures() after you're done with the plots
    to properly free memory.
    
    Args:
        data_path_template (str): Template string for data file paths with {shot:05d} format
        shot_range (range): Range of shot numbers to plot
        n_cols (int): Number of windows to arrange horizontally
        data_processor (Callable, optional): Function to process data before plotting
        monitor_idx (Optional[int]): Monitor index to display plots on. If None, will prompt user.
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
        
        plt.tight_layout()
        
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

if __name__ == "__main__":

    data_path = r"E:\x-ray\20241029\C3--E-ring-wt-Tungsten2mm-xray--{shot:05d}.trc"

    figs = plot_shots_grid(data_path_template=data_path, shot_range=range(0, 5))

