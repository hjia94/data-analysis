'''
This script is used to for different plotting functions.
'''

import tkinter as tk
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from typing import Callable, Optional
from read_scope_data import read_trc_data

def plot_shots_grid(
    data_path_template: str,
    shot_range: range,
    n_cols: int = 4,
    data_processor: Optional[Callable] = None,
    monitor_idx: Optional[int] = None
) -> None:
    """
    Plot multiple shots in a grid layout on a specified monitor.
    
    Args:
        data_path_template (str): Template string for data file paths with {shot:05d} format
        shot_range (range): Range of shot numbers to plot
        n_cols (int): Number of columns in the grid layout
        data_processor (Callable, optional): Function to process data before plotting
        monitor_idx (Optional[int]): Monitor index to display plots on. If None, will prompt user.
    """
    # Get monitor information
    monitors = get_monitors()
    
    # If monitor_idx not provided, show available monitors and prompt user
    if monitor_idx is None:
        print("\nAvailable monitors:")
        for i, m in enumerate(monitors):
            print(f"Monitor {i}: {m.width}x{m.height} at position ({m.x}, {m.y})")
        monitor_idx = int(input("\nEnter the monitor number to display plots on: "))
    
    target_monitor = monitors[monitor_idx]
    
    # Calculate grid layout
    n_shots = len(shot_range)
    n_rows = (n_shots + n_cols - 1) // n_cols  # Ceiling division
    
    # Calculate window sizes
    window_width = target_monitor.width // n_cols
    window_height = target_monitor.height // n_rows
    
    # Base position for windows
    base_x = target_monitor.x
    base_y = target_monitor.y
    
    for i, shot in enumerate(shot_range):
        # Create new figure
        fig = plt.figure(figsize=(8, 6))
        
        # Calculate window position
        row = i // n_cols
        col = i % n_cols
        x_pos = base_x + (col * window_width)
        y_pos = base_y + (row * window_height)
        
        # Read data
        data_path = data_path_template.format(shot=shot)
        data, tarr = read_trc_data(data_path, False)
        
        # Process data if processor provided
        if data_processor:
            data = data_processor(data)
        
        # Plot data
        plt.plot(tarr, -data)
        plt.title(f'Shot {shot}')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        
        # Position window
        mngr = plt.get_current_fig_manager()
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
    
    # Create a small tkinter window to keep the plots alive without blocking
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter window
    root.quit()  # Allow the script to terminate while keeping plots open 