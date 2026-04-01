"""
Emissive Probe Data Analysis for Mar2026 Experiment

This module processes and analyzes emissive probe measurements from the March 2026 LAPD
experiment. The emissive probe is a diagnostic that measures floating potential as a proxy
for plasma potential by thermionic emission.

Functionality:
    - Reads raw emissive probe voltage data (Board 4, Channel 8) from HDF5 files
    - Averages over multiple shots to reduce noise
    - Performs spatial interpolation and 2D mapping
    - Generates time-series plots of plasma potential at specific probe locations
    - Produces 2D spatial maps of plasma potential at different times
    - Saves intermediate data in NPZ format for efficient re-analysis

Data Flow:
    1. Load HDF5 file with digitizer configuration and probe motion data
    2. Extract raw emissive probe signal and convert to voltage (V)
    3. Average over shots to improve signal quality
    4. Save to NPZ for efficient reloading
    5. Generate plots: individual traces and 2D spatial maps

Author: Data analysis pipeline for LAPD Mar2026 campaign
"""

import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from bapsflib import lapd

import read_hdf5 as rh


def get_emissive_data(f, adc, npos, nshot):
    """
    Extract emissive probe voltage data from HDF5 file.
    
    The emissive probe measures floating potential via thermionic emission.
    Data is read from Board 4, Channel 8 and scaled by a calibration factor of 40.
    
    Parameters
    ----------
    f : bapsflib.lapd.File
        Open HDF5 file object
    adc : str
        ADC identifier for the digitizer
    npos : int
        Number of probe positions (spatial locations)
    nshot : int
        Number of shots per position
    
    Returns
    -------
    tuple
        - tarr : np.ndarray
            Time array [s], shape (n_time_samples,)
        - Vp_arr : np.ndarray
            Emissive probe voltage array [V], shape (npos, nshot, n_time_samples)
            Each trace is scaled by 40 (calibration factor)
    """
    # Board 4, Channel 8: Emissive probe voltage signal
    # Raw ADC counts are scaled by 40 to convert to voltage
    data, tarr = rh.read_data(f, 4, 8, index_arr=slice(npos*nshot), adc=adc)
    Vp_arr = data['signal'].reshape((npos, nshot, -1)) * 40  # Scale factor: 40
    
    return tarr, Vp_arr


def save_emissive_data(ifn, save_path):
    """
    Load emissive probe data from HDF5 file and save processed data to NPZ.
    
    This function reads the HDF5 file, extracts all necessary information
    (digitizer config, probe motion, emissive voltage data), and saves it
    to an NPZ file for efficient reloading.
    
    Parameters
    ----------
    ifn : str
        Input HDF5 filename
    save_path : str
        Output NPZ file path where data will be saved
    """
    with lapd.File(ifn) as f:
        # Display file information
        rh.show_info(f)
        
        # Read digitizer configuration
        adc, digi_dict = rh.read_digitizer_config(f)
        
        # Read probe motion (spatial positions)
        pos_dict, xpos, ypos, zpos, npos, nshot = rh.read_bmotion_probe_motion(f)
        key = list(pos_dict.keys())[0]
        pos_array = pos_dict[key]
        
        # Read emissive probe voltage data
        tarr, Vp_arr = get_emissive_data(f, adc, npos, nshot)
    
    # Save all data to NPZ file for later use
    np.savez(save_path, Vp_arr=Vp_arr, tarr=tarr, xpos=xpos, ypos=ypos, 
             zpos=zpos, npos=npos, nshot=nshot, pos_array=pos_array)
    print(f"Emissive probe data saved to: {save_path}")


def load_emissive_data(save_path):
    """
    Load previously saved emissive probe data from NPZ file.
    
    Parameters
    ----------
    save_path : str
        Path to NPZ file containing saved emissive probe data
    
    Returns
    -------
    tuple
        - Vp_arr : np.ndarray
            Emissive probe voltage array [V], shape (npos, nshot, n_time_samples)
        - Vp_arr_avg : np.ndarray
            Shot-averaged voltage array [V], shape (npos, n_time_samples)
        - tarr : np.ndarray
            Time array [s]
        - xpos : np.ndarray
            X positions of probe locations [cm]
        - ypos : np.ndarray
            Y positions of probe locations [cm]
        - pos_array : np.ndarray
            Full position array for each measurement
        - npos : int
            Number of probe positions
        - nshot : int
            Number of shots per position
    """
    data = np.load(save_path)
    Vp_arr = data['Vp_arr']
    tarr = data['tarr']
    xpos = data['xpos']
    ypos = data['ypos']
    pos_array = data['pos_array']
    npos = int(data['npos'])
    nshot = int(data['nshot'])
    
    # Average over shots to reduce noise
    Vp_arr_avg = np.mean(Vp_arr, axis=1)
    
    return Vp_arr, Vp_arr_avg, tarr, xpos, ypos, pos_array, npos, nshot


def plot_individual_traces(Vp_arr, tarr, pos_array, location_indices=None):
    """
    Plot emissive probe voltage traces at selected spatial locations.
    
    Shows how the floating potential varies with time at different probe
    positions. Useful for identifying temporal variations and anomalies.
    
    Parameters
    ----------
    Vp_arr : np.ndarray
        Emissive probe voltage array [V], shape (npos, nshot, n_time_samples)
    tarr : np.ndarray
        Time array [s]
    pos_array : np.ndarray
        Position descriptors for each probe location
    location_indices : list, optional
        Indices of locations to plot. Default: [0, 100, 200, 300, 400]
    """
    if location_indices is None:
        location_indices = [0, 100, 200, 300, 400]
    
    plt.figure(figsize=(12, 6))
    for i in location_indices:
        # Take first shot from each location
        plt.plot(tarr, Vp_arr[i, 0, :], label=f"Pos {i}: {pos_array[i]}", linewidth=2)
    
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Floating Potential [V]")
    plt.title("Emissive Probe: Plasma Potential vs Time at Selected Locations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spatial_maps(Vp_arr_avg, tarr, xpos, ypos, npos, time_indices=None):
    """
    Generate 2D spatial maps of emissive probe signal at selected times.
    
    Creates a grid of 2D heatmaps showing the spatial distribution of floating
    potential at different times during the shot. Useful for visualizing wave
    propagation or spatial structures.
    
    Parameters
    ----------
    Vp_arr_avg : np.ndarray
        Shot-averaged voltage array [V], shape (npos, n_time_samples)
    tarr : np.ndarray
        Time array [s]
    xpos : np.ndarray
        X positions of probe locations [cm]
    ypos : np.ndarray
        Y positions of probe locations [cm]
    npos : int
        Number of probe positions
    time_indices : list, optional
        Time indices to plot. Default: [16500, 23000, 38500, 82500]
    """
    if time_indices is None:
        time_indices = [16500, 23000, 38500, 82500]
    
    # Define spatial extent for image
    extent = (min(xpos), max(xpos), min(ypos), max(ypos))
    
    # Determine grid shape from unique x and y positions
    grid_shape = (len(np.unique(xpos)), len(np.unique(ypos)))
    
    # Create 2x2 subplot grid for 4 time snapshots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, tndx in enumerate(time_indices):
        # Convert time index to milliseconds (with offset for bias timing)
        t_ms = tarr[tndx] * 1e3 + 14 - 15.5
        
        # Average over ±100 indices around the selected time to smooth
        Vp_avg = np.mean(Vp_arr_avg[:, tndx-100:tndx+101], axis=1)
        
        # Reshape 1D probe data into 2D grid
        Vp_2d = Vp_avg.reshape(grid_shape)
        
        # Plot on appropriate subplot
        ax = axs.flat[i]
        ax.set_title(f't = {t_ms:.2f} ms', fontsize=14, fontweight='bold')
        
        # Create heatmap with rainbow colormap
        im = ax.imshow(Vp_2d, origin='lower', cmap=plt.cm.rainbow, 
                       extent=extent, interpolation='gaussian', 
                       vmin=-5, vmax=25)
        ax.set_xlabel('X Position [cm]')
        ax.set_ylabel('Y Position [cm]')
        ax.set_aspect('equal')
    
    # Add single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Floating Potential [V]', fontsize=12)
    
    fig.suptitle('Emissive Probe: 2D Spatial Maps of Plasma Potential', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    plt.show()


# ===============================================================================
# Main execution block
# ===============================================================================

if __name__ == '__main__':
    
    # Specify input HDF5 file
    ifn = r"D:\data\LAPD\Mar26-data\10-emiss-xyplane-coarse-bias.hdf5"
    
    # Extract directory and run number from filename
    data_dir = os.path.dirname(ifn)
    run_num = os.path.basename(ifn).split('-')[0]  # Extract "10" from filename
    
    # Define output file path
    save_path = os.path.join(data_dir, f"{run_num}-emissive-data.npz")
    
    # =========================================================================
    # Option 1: Load from HDF5 and save to NPZ (uncomment to run)
    # =========================================================================
    # Uncomment the line below if you need to reload data from the original HDF5
    # save_emissive_data(ifn, save_path)
    
    # =========================================================================
    # Option 2: Load from saved NPZ file (faster, recommended after first run)
    # =========================================================================
    print(f"Loading emissive probe data from: {save_path}")
    Vp_arr, Vp_arr_avg, tarr, xpos, ypos, pos_array, npos, nshot = load_emissive_data(save_path)
    
    print(f"\nData loaded successfully!")
    print(f"  Array shape (npos × nshot × time_samples): {Vp_arr.shape}")
    print(f"  Time range: {tarr[0]:.6f} to {tarr[-1]:.6f} s")
    print(f"  Spatial coverage: {len(np.unique(xpos))} × {len(np.unique(ypos))} grid")
    
    # =========================================================================
    # Generate plots
    # =========================================================================
    
    # Plot 1: Individual time traces at selected probe locations
    print("\nGenerating plot 1: Individual traces at selected locations...")
    plot_individual_traces(Vp_arr, tarr, pos_array)
    
    # Plot 2: 2D spatial maps at different times
    print("Generating plot 2: 2D spatial maps at selected times...")
    plot_spatial_maps(Vp_arr_avg, tarr, xpos, ypos, npos)