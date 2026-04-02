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
        pos_dict, xpos, ypos, zpos, npos, nshot = rh.read_bmotion_probe_motion(f)
        key = list(pos_dict.keys())[0]
        pos_array = pos_dict[key]
        
        tarr, Vxp_arr, Vxm_arr, Vyp_arr, Vym_arr = get_mach_data(f, adc, npos, nshot)
        print('Applying Gaussian smoothing to raw data')
        for i in tqdm(range(npos), desc="Smoothing (Gaussian)"):
            Vxp_arr[i] = ndimage.gaussian_filter1d(Vxp_arr[i], sigma=50, axis=-1)
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
    
    Creates a 4 × num_parts grid showing spatial maps for each channel (Vx+, Vx-, Vy+, Vy-)
    averaged over time bins.
    
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

    # Calculate time bin size
    part_len = nt // num_parts

    # Define spatial extent
    extent = (xpos.min(), xpos.max(), ypos.min(), ypos.max())

    # Calculate global min/max for consistent colormapping
    vmin = 0
    vmax = max(heat_Vxp.max(), heat_Vxm.max(), heat_Vyp.max(), heat_Vym.max())

    # 4 rows (Vx+, Vx-, Vy+, Vy-), num_parts columns
    fig, axs = plt.subplots(4, num_parts, figsize=(1.5*num_parts, 10), squeeze=False)

    channel_data = [
        (heat_Vxp, 'Vx+'),
        (heat_Vxm, 'Vx-'),
        (heat_Vyp, 'Vy+'),
        (heat_Vym, 'Vy-')
    ]

    for ch_idx, (heat_map, ch_label) in enumerate(channel_data):
        for part in range(num_parts):
            start_idx = part * part_len
            end_idx = (part + 1) * part_len if part < num_parts - 1 else nt
            t_center = tarr[start_idx:end_idx].mean()
            ch_avg = heat_map[:, :, start_idx:end_idx].mean(axis=2)

            ax = axs[ch_idx, part]
            im = ax.imshow(ch_avg, origin='lower', cmap='magma',
                          extent=extent, interpolation='gaussian',
                          vmin=vmin, vmax=vmax)
            
            # Title only on first row
            if ch_idx == 0:
                ax.set_title(f"{t_center*1e3:.2f} ms", fontsize=8)
            
            # Y-label on first column
            if part == 0:
                ax.set_ylabel(ch_label, fontsize=11, weight='bold')

    # X labels only on bottom row
    for part in range(num_parts):
        axs[3, part].set_xlabel('X [cm]', fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12,
                        wspace=0.10, hspace=0.15)

    # horizontal shared colorbar at bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.70, 0.015])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                label='Voltage amplitude (V)')

    fig.suptitle(f'Mach Probe: All 4 Channels Averaged Over {num_parts} Time Bins',
                fontsize=14, fontweight='bold')

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