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
import matplotlib.pyplot as plt
from scipy import signal, ndimage, interpolate
from scipy.fft import next_fast_len
from tqdm import tqdm

sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

from bapsflib import lapd
import read_hdf5 as rh
from data_analysis_utils import butter_bandpass


def get_mach_data(f, adc, npos, nshot):
    """
    Extract Mach probe voltage data from HDF5 file.
    
    The Mach probe measures perpendicular ion saturation current in two directions.
    Data is read from:
    - Board 1, Channel 1: Vx component
    - Board 4, Channel 6: Vy component
    
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
        - Vx_arr : np.ndarray
            X-component Mach probe voltage array, shape (npos, nshot, n_time_samples)
        - Vy_arr : np.ndarray
            Y-component Mach probe voltage array, shape (npos, nshot, n_time_samples)
    """
    data, tarr = rh.read_data(f, 1, 1, index_arr=slice(npos * nshot), adc=adc)
    Vx_arr = data['signal'].reshape((npos, nshot, -1))
    
    data, tarr = rh.read_data(f, 4, 6, index_arr=slice(npos * nshot), adc=adc)
    Vy_arr = data['signal'].reshape((npos, nshot, -1))
    
    return tarr, Vx_arr, Vy_arr


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
        pos_dict, xpos, ypos, zpos, npos, nshot = rh.read_bmotion_probe_motion(f)
        key = list(pos_dict.keys())[0]
        pos_array = pos_dict[key]
        
        tarr, Vx_arr, Vy_arr = get_mach_data(f, adc, npos, nshot)
        print('Applying Gaussian smoothing to raw data')
        Vx_arr = ndimage.gaussian_filter1d(Vx_arr, sigma=50, axis=-1)
        Vy_arr = ndimage.gaussian_filter1d(Vy_arr, sigma=50, axis=-1)
    
    # Save all data to NPZ file for later use
    np.savez(save_path, Vx_arr=Vx_arr, Vy_arr=Vy_arr, tarr=tarr, 
             xpos=xpos, ypos=ypos, zpos=zpos, npos=npos, nshot=nshot, pos_array=pos_array)
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
    Vx_arr = data['Vx_arr']
    Vy_arr = data['Vy_arr']
    tarr = data['tarr']
    xpos = data['xpos']
    ypos = data['ypos']
    pos_array = data['pos_array']
    npos = int(data['npos'])
    nshot = int(data['nshot'])
    
    return Vx_arr, Vy_arr, tarr, xpos, ypos, pos_array, npos, nshot

def process_mach_envelopes(Vx_arr, Vy_arr, tarr, npos, fs=1.0e6, lowcut=500.0, highcut=12000.0, filter_order=4, median_window=5001, rms_window=500, downsample_factor=10):
    """
    Extracts amplitude envelopes of oscillations from Vx and Vy signal arrays 
    using optimized baseline subtraction and Moving RMS envelopes.
    
    Assumes Vx_arr and Vy_arr have the shape (npos, n_shots, n_timepoints)
    """
    
    n_shots = Vx_arr.shape[1]
    n_timepoints = Vx_arr.shape[2]
    
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
    
    # Initialize output heatmaps
    heatmap_Vx = np.zeros((npos, downsampled_timepoints))
    heatmap_Vy = np.zeros((npos, downsampled_timepoints))
    
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
        heatmap_Vx[i, :] = process_component(Vx_arr[i, :, :])
        heatmap_Vy[i, :] = process_component(Vy_arr[i, :, :])
        
    # ==========================================
    # 4. Process Time Array & Return
    # ==========================================
    # Downsample the time array to match the width of the heatmaps
    tarr_reshaped = tarr[:valid_time_length].reshape(-1, downsample_factor)
    tarr_downsampled = np.mean(tarr_reshaped, axis=1)

    return {
        'heatmap_Vx': heatmap_Vx,
        'heatmap_Vy': heatmap_Vy,
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
    
    Vx_arr, Vy_arr, tarr, xpos, ypos, pos_array, npos, nshot = load_mach_data(raw_path)
    
    # Process envelopes
    try:
        result = process_mach_envelopes(Vx_arr, Vy_arr, tarr, npos, **process_kwargs)
    except KeyboardInterrupt:
        print("Processing interrupted by user ctrl+c.")
        return
    except Exception as e:
        print(f"Error during envelope processing: {e}")
        return
    
    # Save processed data
    np.savez(save_path,
             heatmap_Vx=result['heatmap_Vx'],
             heatmap_Vy=result['heatmap_Vy'],
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
        'heatmap_Vx': data['heatmap_Vx'],
        'heatmap_Vy': data['heatmap_Vy'],
        'tarr_downsampled': data['tarr_downsampled'],
        'xpos': data['xpos'],
        'ypos': data['ypos'],
        'npos': int(data['npos'])
    }


def plot_mach_heatmap(result, num_parts=10):

    xpos = result['xpos']
    ypos = result['ypos']
    tarr = result['tarr_downsampled']
    nx = len(xpos)
    ny = len(ypos)
    nt = len(tarr)
    heat_map_Vx = result['heatmap_Vx'].reshape(ny, nx, nt)
    heat_map_Vy = result['heatmap_Vy'].reshape(ny, nx, nt)

    # Calculate time bin size
    part_len = nt // num_parts

    # Define spatial extent
    extent = (xpos.min(), xpos.max(), ypos.min(), ypos.max())

    # one shared color range for both fields
    vmin = 0
    vmax = 0.02

    # 2 rows (Vx, Vy), num_parts columns (long skinny)
    fig, axs = plt.subplots(2, num_parts, figsize=(1.5*num_parts, 6), squeeze=False)

    for part in range(num_parts):
        start_idx = part * part_len
        end_idx = (part + 1) * part_len if part < num_parts - 1 else nt
        t_center = tarr[start_idx:end_idx].mean()
        vx_avg = heat_map_Vx[:, :, start_idx:end_idx].mean(axis=2)
        vy_avg = heat_map_Vy[:, :, start_idx:end_idx].mean(axis=2)

        ax_vx = axs[0, part]
        ax_vy = axs[1, part]

        im = ax_vx.imshow(vx_avg, origin='lower', cmap='magma',
                        extent=extent, interpolation='gaussian',
                        vmin=vmin, vmax=vmax)
        ax_vx.set_title(f"{t_center*1e3:.2f} ms", fontsize=8)

        ax_vy.imshow(vy_avg, origin='lower', cmap='magma',
                    extent=extent, interpolation='gaussian',
                    vmin=vmin, vmax=vmax)


    # shared row labels
    fig.text(0.01, 0.72, "Vx", va="center", ha="left", fontsize=12, weight="bold")
    fig.text(0.01, 0.28, "Vy", va="center", ha="left", fontsize=12, weight="bold")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.15,
                        wspace=0.12, hspace=0.12)

    # horizontal shared colorbar at bottom
    cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                label='Voltage amplitude (V)')

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
    # Option 1: Process from HDF5 and save envelopes
    # =========================================================================
    # save_mach_envelope_data(ifn, envelope_save_path)
    
    # =========================================================================
    # Option 2: Load previously processed envelopes and visualize
    # =========================================================================
    result = load_mach_envelope_data(envelope_save_path)
    plot_mach_heatmap(result, num_parts=10)