"""
Mach Probe Data Analysis for Mar2026 Experiment

This module processes and analyzes Mach probe measurements from the March 2026 LAPD
experiment. The Mach probe is a diagnostic that measures plasma flow velocity via
directional ion saturation current.

Functionality:
    - Reads raw Mach probe voltage data (Board 1 Channel 1, Board 4 Channel 6) from HDF5 files
    - Performs non-linear baseline subtraction and bandpass filtering
    - Extracts oscillation envelopes using Hilbert transform
    - Averages over multiple shots to reduce noise
    - Generates heatmaps showing oscillation amplitude across spatial positions and time
    - Saves intermediate data in NPZ format for efficient re-analysis

Data Flow:
    1. Load HDF5 file with digitizer configuration and probe motion data
    2. Extract raw Mach probe signals (Vx, Vy channels)
    3. Process signals with baseline subtraction, filtering, and envelope detection
    4. Average envelopes over shots
    5. Downsample for visualization
    6. Save to NPZ for efficient reloading
    7. Generate heatmaps showing amplitude evolution

Processing Pipeline:
    - Median filter baseline subtraction (non-linear)
    - Butterworth bandpass filter (10-50 kHz)
    - Hilbert transform for envelope detection
    - Shot averaging (envelope-based, prevents phase cancellation)
    - Visualization with downsampling for performance

Author: Data analysis pipeline for LAPD Mar2026 campaign
"""

import nt
import os
import re
import sys
from jellyfish import nysiis
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import signal, ndimage, interpolate
from scipy.fft import next_fast_len
from tqdm import tqdm

sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

from bapsflib import lapd
import read_hdf5_bapsflib as rh
from data_analysis_utils import butter_bandpass


def get_mach_data(f, adc, npos, nshot):
    """
    Extract Mach probe voltage data from HDF5 file.
    
    The Mach probe measures perpendicular ion saturation current in two directions.
    Data is read from:
    - Board 1, Channel 1: Vx+ component
    - Board 1, Channel 2: Vx- component
    - Board 4, Channel 6: Vy component
    - Board 4, Channel 7: Vy- component
    
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
        - Vxp_arr : np.ndarray
            X+ component voltage array, shape (npos, nshot, n_time_samples)
        - Vxm_arr : np.ndarray
            X- component voltage array, shape (npos, nshot, n_time_samples)
        - Vyp_arr : np.ndarray
            Y+ component voltage array, shape (npos, nshot, n_time_samples)
        - Vym_arr : np.ndarray
            Y- component voltage array, shape (npos, nshot, n_time_samples)
    """
    # Board 1, Channel 1: Vx+ component
    data, tarr = rh.read_data(f, 1, 1, index_arr=slice(npos * nshot), adc=adc)
    Vxp_arr = data['signal'].reshape((npos, nshot, -1))
    
    # Board 1, Channel 2: Vx- component
    data, tarr = rh.read_data(f, 1, 2, index_arr=slice(npos * nshot), adc=adc)
    Vxm_arr = data['signal'].reshape((npos, nshot, -1))
    
    # Board 4, Channel 6: Vy component
    data, tarr = rh.read_data(f, 4, 6, index_arr=slice(npos * nshot), adc=adc)
    Vyp_arr = data['signal'].reshape((npos, nshot, -1))
    
    # Board 4, Channel 7: Vy- component
    data, tarr = rh.read_data(f, 4, 7, index_arr=slice(npos * nshot), adc=adc)
    Vym_arr = data['signal'].reshape((npos, nshot, -1))
    
    return tarr, Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr


def save_mach_data(ifn, save_path):
    """
    Load Mach probe data from HDF5 file and save processed data to NPZ.
    
    This function reads the HDF5 file, extracts all necessary information
    (digitizer config, probe motion, Mach probe voltage data), and saves it
    to an NPZ file for efficient reloading.
    
    Parameters
    ----------
    ifn : str
        Input HDF5 filename
    save_path : str
        Output NPZ file path where data will be saved
    """
    with lapd.File(ifn) as f:
        adc, digi_dict = rh.read_digitizer_config(f)
        pos_dict, xpos, ypos, zpos, npos, nshot = rh.read_probe_motion_bmotion(f)
        key = list(pos_dict.keys())[0]
        pos_array = pos_dict[key]
        
        tarr, Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr = get_mach_data(f, adc, npos, nshot)
        print('Applying Gaussian smoothing to raw data')
        for i in tqdm(range(npos), desc="Smoothing (Gaussian)"):
            Vxp_arr[i] = ndimage.gaussian_filter1d(Vxp_arr[i], sigma=50, axis=-1)
            Vxm_arr[i] = ndimage.gaussian_filter1d(Vxm_arr[i], sigma=50, axis=-1)
            Vyp_arr[i] = ndimage.gaussian_filter1d(Vyp_arr[i], sigma=50, axis=-1)
            Vym_arr[i] = ndimage.gaussian_filter1d(Vym_arr[i], sigma=50, axis=-1)
            Vxm_arr[i] = ndimage.gaussian_filter1d(Vxm_arr[i], sigma=50, axis=-1)
            Vyp_arr[i] = ndimage.gaussian_filter1d(Vyp_arr[i], sigma=50, axis=-1)
            Vym_arr[i] = ndimage.gaussian_filter1d(Vym_arr[i], sigma=50, axis=-1)
    
    # Save all data to NPZ file for later use
    np.savez(save_path, Vxp_arr=Vxp_arr, Vxm_arr=Vxm_arr, Vyp_arr=Vyp_arr, Vym_arr=Vym_arr,
             tarr=tarr, xpos=xpos, ypos=ypos, zpos=zpos, npos=npos, nshot=nshot, pos_array=pos_array)
    print(f"Mach probe raw data saved to: {save_path}")


def load_mach_data(save_path):
    """
    Load previously saved Mach probe raw data from NPZ file.
    
    Parameters
    ----------
    save_path : str
        Path to NPZ file containing saved Mach probe raw data
    
    Returns
    -------
    tuple
        - Vx_arr : np.ndarray
            X-component voltage array, shape (npos, nshot, n_time_samples)
        - Vy_arr : np.ndarray
            Y-component voltage array, shape (npos, nshot, n_time_samples)
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
    Vxp_arr = data['Vxp_arr']
    Vxm_arr = data['Vxm_arr']
    Vyp_arr = data['Vyp_arr']
    Vym_arr = data['Vym_arr']
    tarr = data['tarr']
    xpos = data['xpos']
    ypos = data['ypos']
    pos_array = data['pos_array']
    npos = int(data['npos'])
    nshot = int(data['nshot'])
    
    return Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr, tarr, xpos, ypos, pos_array, npos, nshot

def process_mach_envelopes(Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr, tarr, npos, fs=1.0e6, lowcut=500.0, highcut=12000.0, filter_order=4, median_window=5001, rms_window=500, downsample_factor=10):
    """
    Extracts amplitude envelopes of oscillations from all 4 Mach probe channels.
    
    Uses optimized baseline subtraction and Moving RMS envelopes.
    Assumes all input arrays have the shape (npos, n_shots, n_timepoints)
    
    Returns envelopes for all 4 channels: Vx+, Vx-, Vy+, Vy-
    """
    
    n_shots = Vxp_arr.shape[1]
    n_timepoints = Vxp_arr.shape[2]
    
    # ==========================================
    # 1. Pipeline Setup & Pre-calculations
    # ==========================================
    # Stable Bandpass filter design (SOS)
    nyq = 0.5 * fs
    sos = signal.butter(filter_order, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
    
    # Baseline Decimation Setup (Fixed to 50 for max speed)
    baseline_dec_factor = 50 
    x_orig = np.arange(n_timepoints)
    x_dec = np.arange(0, n_timepoints, baseline_dec_factor)
    
    # Scale median window for the decimated array
    small_window = median_window // baseline_dec_factor
    if small_window % 2 == 0: 
        small_window += 1 # Ensure odd integer
        
    # Output Downsampling Setup
    downsampled_timepoints = n_timepoints // downsample_factor
    valid_time_length = downsampled_timepoints * downsample_factor
    
    # Initialize output heatmaps for all 4 channels
    heatmap_Vxp = np.zeros((npos, downsampled_timepoints))
    heatmap_Vxm = np.zeros((npos, downsampled_timepoints))
    heatmap_Vyp = np.zeros((npos, downsampled_timepoints))
    heatmap_Vym = np.zeros((npos, downsampled_timepoints))
    
    # ==========================================
    # 2. Core Processing Function
    # ==========================================
    def process_component(shots):
        """Processes a (n_shots, n_timepoints) array for a single position."""
        
        # STEP A: Decimate & Interpolate Baseline
        shots_dec = shots[:, ::baseline_dec_factor]
        baseline_dec = ndimage.median_filter(shots_dec, size=(1, small_window))
        baseline = interpolate.interp1d(x_dec, baseline_dec, axis=-1, fill_value="extrapolate")(x_orig)
        flattened_shots = shots - baseline
        
        # STEP B: Stable Bandpass Filter (Prevents numerical explosion)
        filtered_shots = signal.sosfiltfilt(sos, flattened_shots, axis=-1)
        
        # STEP C: Moving RMS Envelope Calculation
        squared_shots = filtered_shots ** 2
        
        # uniform_filter1d acts as a blazing fast moving average
        mean_squares = ndimage.uniform_filter1d(squared_shots, size=rms_window, axis=-1)
        
        # Take the square root and multiply by sqrt(2) so it matches peak amplitude
        rms_envelopes = np.sqrt(mean_squares) * np.sqrt(2)
        
        # STEP D: Downsample EACH shot independently
        reshaped_envelopes = rms_envelopes[:, :valid_time_length].reshape(n_shots, -1, downsample_factor)
        downsampled_envelopes = np.mean(reshaped_envelopes, axis=-1)
        
        # STEP E: Average across the 4 shots as the very last step
        final_averaged_envelope = np.mean(downsampled_envelopes, axis=0)
        
        return final_averaged_envelope

    # ==========================================
    # 3. Main Loop
    # ==========================================
    print(f"Processing {npos} positions...")
    for i in tqdm(range(npos), desc="Extracting Envelopes"):
        heatmap_Vxp[i, :] = process_component(Vxp_arr[i, :, :])
        heatmap_Vxm[i, :] = process_component(Vxm_arr[i, :, :])
        heatmap_Vyp[i, :] = process_component(Vyp_arr[i, :, :])
        heatmap_Vym[i, :] = process_component(Vym_arr[i, :, :])
        
    # ==========================================
    # 4. Process Time Array & Return
    # ==========================================
    # Downsample the time array to match the width of the heatmaps
    tarr_reshaped = tarr[:valid_time_length].reshape(-1, downsample_factor)
    tarr_downsampled = np.mean(tarr_reshaped, axis=1)

    return {
        'heatmap_Vxp': heatmap_Vxp,
        'heatmap_Vxm': heatmap_Vxm,
        'heatmap_Vyp': heatmap_Vyp,
        'heatmap_Vym': heatmap_Vym,
        'tarr_downsampled': tarr_downsampled,
        'parameters': {
            'fs': fs,
            'lowcut': lowcut,
            'highcut': highcut,
            'filter_order': filter_order,
            'median_window': median_window,
            'rms_window': rms_window,
            'downsample_factor': downsample_factor
        }
    }


def save_mach_envelope_data(ifn, save_path, **process_kwargs):
    """
    Load Mach probe data, process envelopes, and save to NPZ.
    
    Parameters
    ----------
    ifn : str
        Input HDF5 filename
    save_path : str
        Output NPZ file path where processed envelope data will be saved
    **process_kwargs : dict
        Keyword arguments to pass to process_mach_envelopes()
    """
    # Load raw data
    raw_path = save_path.replace('.npz', '_raw.npz')
    if not os.path.exists(raw_path):
        save_mach_data(ifn, raw_path)
    
    Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr, tarr, xpos, ypos, pos_array, npos, nshot = load_mach_data(raw_path)
    
    # Process envelopes
    try:
        result = process_mach_envelopes(Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr, tarr, npos, **process_kwargs)
    except KeyboardInterrupt:
        print("Processing interrupted by user ctrl+c.")
        return
    except Exception as e:
        print(f"Error during envelope processing: {e}")
        return
    
    # Save processed data
    np.savez(save_path,
             heatmap_Vxp=result['heatmap_Vxp'],
             heatmap_Vxm=result['heatmap_Vxm'],
             heatmap_Vyp=result['heatmap_Vyp'],
             heatmap_Vym=result['heatmap_Vym'],
             tarr_downsampled=result['tarr_downsampled'],
             xpos=xpos, ypos=ypos, npos=npos,
             fs=result['parameters']['fs'],
             lowcut=result['parameters']['lowcut'],
             highcut=result['parameters']['highcut'])
    
    print(f"Mach probe envelope data saved to: {save_path}")


def load_mach_envelope_data(save_path):
    """
    Load previously saved Mach probe envelope data from NPZ file.
    
    Parameters
    ----------
    save_path : str
        Path to NPZ file containing saved Mach probe envelope data
    
    Returns
    -------
    dict
        Dictionary containing heatmaps and associated metadata
    """
    data = np.load(save_path)
    return {
        'heatmap_Vxp': data['heatmap_Vxp'],
        'heatmap_Vxm': data['heatmap_Vxm'],
        'heatmap_Vyp': data['heatmap_Vyp'],
        'heatmap_Vym': data['heatmap_Vym'],
        'tarr_downsampled': data['tarr_downsampled'],
        'xpos': data['xpos'],
        'ypos': data['ypos'],
        'npos': int(data['npos'])
    }


def plot_mach_heatmap(result, num_parts=10):
    """
    Generate heatmap visualization of all 4 Mach probe channels.
    
    Creates a num_parts column grid showing spatial maps of the summed signal
    (Vx+ + Vx- + Vy+ + Vy-) averaged over time bins, with contour overlays.
    
    Parameters
    ----------
    result : dict
        Dictionary from load_mach_envelope_data containing heatmaps and metadata
    num_parts : int
        Number of time bins (default: 10)
    """
    xpos = result['xpos']
    ypos = result['ypos']
    tarr = result['tarr_downsampled']
    nx = len(xpos)
    ny = len(ypos)
    nt = len(tarr)
    
    # Reshape all 4 channels to 3D
    heat_Vxp = result['heatmap_Vxp'].reshape(ny, nx, nt)
    heat_Vxm = result['heatmap_Vxm'].reshape(ny, nx, nt)
    heat_Vyp = result['heatmap_Vyp'].reshape(ny, nx, nt)
    heat_Vym = result['heatmap_Vym'].reshape(ny, nx, nt)

    # Sum all 4 channels
    heat_sum = heat_Vxp + heat_Vxm + heat_Vyp + heat_Vym

    # Calculate time bin size
    part_len = nt // num_parts

    # Define spatial extent
    extent = (xpos.min(), xpos.max(), ypos.min(), ypos.max())

    # Calculate global min/max for consistent colormapping
    vmin = 0
    vmax = heat_sum.max()

    # 1 row, num_parts columns
    fig, axs = plt.subplots(1, num_parts, figsize=(1.5*num_parts, 4), squeeze=False)

    for part in range(num_parts):
        start_idx = part * part_len
        end_idx = (part + 1) * part_len if part < num_parts - 1 else nt
        t_center = tarr[start_idx:end_idx].mean()
        ch_avg = heat_sum[:, :, start_idx:end_idx].mean(axis=2)

        ax = axs[0, part]
        im = ax.imshow(ch_avg, origin='lower', cmap='magma',
                      extent=extent, interpolation='gaussian',
                      vmin=vmin, vmax=vmax)
        
        # Add contours
        X, Y = np.meshgrid(xpos, ypos)
        ax.contour(X, Y, ch_avg, levels=8, colors='white', alpha=0.4, linewidths=0.5)
        
        ax.set_title(f"{t_center*1e3:.2f} ms", fontsize=8)
        ax.set_xlabel('X [cm]', fontsize=8)
        if part == 0:
            ax.set_ylabel('Y [cm]', fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.15,
                        wspace=0.25, hspace=0.15)

    # Colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.75])
    fig.colorbar(im, cax=cbar_ax, label='Summed Voltage Amplitude (V)')

    fig.suptitle('Mach Probe: Sum of Vxp + Vxm + Vyp + Vym', fontsize=14, fontweight='bold')

    plt.show()


def plot_combined_ne_mach(mach_result, iv_ne_arr, iv_t_ls, iv_xpos=None, iv_ypos=None, num_parts=10):
    """
    Plot electron density from IV probe as imshow with Mach probe heatmap as contours.
    Divides both datasets into num_parts time bins and averages within each bin.
    
    Both datasets must have the same spatial grid (same xpos, ypos).
    
    Parameters
    ----------
    mach_result : dict
        Dictionary from load_mach_envelope_data containing heatmaps and metadata
    iv_ne_arr : np.ndarray
        Electron density array from IV analysis (n_locs, n_sweeps)
    iv_t_ls : np.ndarray
        Time array for IV sweeps [s]
    iv_xpos : np.ndarray, optional
        X positions for IV probe. If None, uses mach_result['xpos']
    iv_ypos : np.ndarray, optional
        Y positions for IV probe. If None, uses mach_result['ypos']
    num_parts : int
        Number of time bins (default: 10)
    """
    # Get mach data
    mach_tarr = mach_result['tarr_downsampled']
    mach_xpos = mach_result['xpos']
    mach_ypos = mach_result['ypos']
    mach_nt = len(mach_tarr)
    
    # Sum all 4 mach channels
    mach_heatmap_sum = (mach_result['heatmap_Vxp'] + mach_result['heatmap_Vxm'] + 
                        mach_result['heatmap_Vyp'] + mach_result['heatmap_Vym'])
    
    nx = len(mach_xpos)
    ny = len(mach_ypos)
    mach_heatmap_3d = mach_heatmap_sum.reshape(ny, nx, mach_nt)
    
    # Normalize mach heatmap to [0, 1] for consistent contour levels
    mach_heatmap_3d_normalized = mach_heatmap_3d
    
    # Spatial extent (use mach positions if IV positions not provided)
    xpos_plot = iv_xpos if iv_xpos is not None else mach_xpos
    ypos_plot = iv_ypos if iv_ypos is not None else mach_ypos
    extent = (xpos_plot.min(), xpos_plot.max(), ypos_plot.min(), ypos_plot.max())
    
    grid_shape = (ny, nx)
    
    # Define time bins (use IV time range)
    iv_t_start = iv_t_ls.min()
    iv_t_end = iv_t_ls.max()
    time_edges = np.linspace(iv_t_start, iv_t_end, num_parts + 1)
    
    # Determine subplot layout (5 columns)
    ncols = 5
    nrows = (num_parts + ncols - 1) // ncols
    
    # Create figure
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 4*nrows), squeeze=False)
    axs = axs.flatten()
    
    im_obj = None  # To store image object for colorbar
    
    # Plot each time bin
    for part in range(num_parts):
        t_start = time_edges[part]
        t_end = time_edges[part + 1]
        t_center = (t_start + t_end) / 2
        
        # Find IV sweeps within this time window
        mask = (iv_t_ls >= t_start) & (iv_t_ls < t_end)
        if part == num_parts - 1:  # Include last edge in final bin
            mask = (iv_t_ls >= t_start) & (iv_t_ls <= t_end)
        
        # Average ne over sweeps in this window
        ne_in_window = iv_ne_arr[:, mask]
        if ne_in_window.shape[1] > 0:
            ne_avg = np.nanmean(ne_in_window, axis=1)
        else:
            ne_avg = np.full(iv_ne_arr.shape[0], np.nan)
        
        ne_2d = ne_avg.reshape(grid_shape)
        
        # Find closest Mach time to center of bin
        mach_time_idx = np.argmin(np.abs(mach_tarr - t_center))
        mach_2d = mach_heatmap_3d_normalized[:, :, mach_time_idx]
        
        # Plot
        ax = axs[part]
        
        # Imshow for mach heatmap
        im_obj = ax.imshow(mach_2d, origin='lower', cmap='hot', extent=extent, 
                           interpolation='gaussian', vmin=0, vmax=0.1)
        
        # Contours for electron density (4 levels with distinct colors for each level)
        contour = ax.contour(ne_2d, extent=extent, colors=['cyan', 'blue', 'orange', 'red'], linewidths=1.2,
                             levels=[1e12, 2e12, 3e12, 4e12])
        
        # Title with time only
        ax.set_title(f'{t_center*1e3:.1f} ms', fontsize=9)
        # ax.set_xticks([])
        # ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_parts, len(axs)):
        axs[idx].axis('off')
    
    # Add contour level legend at top of figure with colored values
    contour_levels = [1e12, 2e12, 3e12, 4e12]
    contour_colors = ['cyan', 'blue', 'orange', 'red']
    fig.text(0.5, 1, 'Electron Density Levels [cm$^{-3}$]:', 
             ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Position colored level values across the top
    x_positions = [0.20, 0.38, 0.56, 0.74]
    for x_pos, level, color in zip(x_positions, contour_levels, contour_colors):
        fig.text(x_pos, 0.95, f'{level:.0e}', color=color, fontsize=11, fontweight='bold')
    
    # Single colorbar at bottom
    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12,
                        wspace=0.15, hspace=0.4)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.70, 0.02])
    fig.colorbar(im_obj, cax=cbar_ax, orientation='horizontal',
                label='Mach Amplitude [V]')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




# ===============================================================================
# Main execution block
# ===============================================================================

if __name__ == '__main__':
    
    ifn = r"D:\data\LAPD\Mar26-data\09-mach-xyplane-bias.hdf5"
    
    # Extract directory and run number from ifn
    data_dir = os.path.dirname(ifn)
    run_num = os.path.basename(ifn).split('-')[0]
    
    # Define file paths
    envelope_save_path = os.path.join(data_dir, f"{run_num}-mach-envelope-data.npz")
    
    # =========================================================================
    # Load Mach probe data (run 09)
    # =========================================================================
    print("Loading Mach probe envelope data...")
    mach_result = load_mach_envelope_data(envelope_save_path)
    
    # =========================================================================
    # Load IV/Plasma data (run 11)
    # =========================================================================
    print("Loading IV/Plasma data...")
    from Mar2026_IV import load_data
    Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls = load_data(data_dir, '11')
    
    # =========================================================================
    # Plot combined: electron density with Mach probe contours
    # =========================================================================
    print("Generating combined plot...")
    plot_combined_ne_mach(mach_result, ne_arr, t_ls, num_parts=10)