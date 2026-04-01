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

import os
import sys
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

def process_mach_envelopes(Vx_arr, Vy_arr, tarr, npos, fs=1.0e6, lowcut=8000.0, highcut=30000.0, filter_order=4, median_window=5001, downsample_factor=10):
    """
    Extracts amplitude envelopes of oscillations from Vx and Vy signal arrays 
    using highly optimized baseline subtraction and Hilbert transforms.
    
    Assumes Vx_arr and Vy_arr have the shape (npos, n_shots, n_timepoints)
    """
    
    n_timepoints = len(tarr)
    
    # ==========================================
    # 1. Pipeline Setup & Pre-calculations
    # ==========================================
    # Bandpass filter design
    nyq = 0.5 * fs
    b, a = signal.butter(filter_order, [lowcut/nyq, highcut/nyq], btype='band')
    
    # Fast FFT length for optimal Hilbert Transform speed
    fast_len = next_fast_len(n_timepoints)
    
    # Baseline Decimation Setup (Fixed to 50 for max speed, independent of imshow downsampling)
    baseline_dec_factor = 100
    x_orig = np.arange(n_timepoints)
    x_dec = np.arange(0, n_timepoints, baseline_dec_factor)
    
    # Scale median window for the decimated array
    small_window = median_window // baseline_dec_factor
    if small_window % 2 == 0: 
        small_window += 1 # Ensure odd integer
        
    # Output Downsampling Setup (for final heatmap rendering)
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
        
        # STEP B: Bandpass Filter
        filtered_shots = signal.filtfilt(b, a, flattened_shots, axis=-1)
        
        # STEP C: Fast Hilbert Transform
        # Calculate with optimized padding, then slice back to original shape
        analytic_signal = signal.hilbert(filtered_shots, N=fast_len, axis=-1)
        amplitude_envelopes = np.abs(analytic_signal[..., :n_timepoints])
        
        # STEP D: Average the Phase-Agnostic Envelopes across shots
        mean_envelope = np.mean(amplitude_envelopes, axis=0)
        
        # STEP E: Downsample for visualization/output
        reshaped_envelope = mean_envelope[:valid_time_length].reshape(-1, downsample_factor)
        downsampled_envelope = np.mean(reshaped_envelope, axis=1)
        
        return downsampled_envelope

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


def plot_mach_heatmap(heatmap_Vx, heatmap_Vy, tarr_downsampled, npos, 
                      title_suffix=''):
    """
    Generate heatmap visualization of Mach probe oscillation envelopes.
    
    Creates two side-by-side heatmaps showing oscillation amplitude evolution
    for both Vx and Vy components across all spatial positions and time.
    
    Parameters
    ----------
    heatmap_Vx : np.ndarray
        X-component heatmap matrix, shape (npos, downsampled_timepoints)
    heatmap_Vy : np.ndarray
        Y-component heatmap matrix, shape (npos, downsampled_timepoints)
    tarr_downsampled : np.ndarray
        Downsampled time array [s]
    npos : int
        Number of probe positions
    title_suffix : str
        Optional suffix to add to the title (e.g., run number)
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    max_time = tarr_downsampled[-1]
    
    # Plot Vx component
    im_vx = axs[0].imshow(heatmap_Vx, aspect='auto', cmap='magma', origin='lower',
                          extent=[0, max_time, 0, npos])
    axs[0].set_xlabel('Time (s)', fontsize=12)
    axs[0].set_ylabel('Position Index', fontsize=12)
    axs[0].set_title(f'Mach Probe Vx Component {title_suffix}', fontsize=14, fontweight='bold')
    cbar_vx = plt.colorbar(im_vx, ax=axs[0])
    cbar_vx.set_label('Envelope Amplitude', fontsize=11)
    
    # Plot Vy component
    im_vy = axs[1].imshow(heatmap_Vy, aspect='auto', cmap='magma', origin='lower',
                          extent=[0, max_time, 0, npos])
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('Position Index', fontsize=12)
    axs[1].set_title(f'Mach Probe Vy Component {title_suffix}', fontsize=14, fontweight='bold')
    cbar_vy = plt.colorbar(im_vy, ax=axs[1])
    cbar_vy.set_label('Envelope Amplitude', fontsize=11)
    
    fig.suptitle(f'Oscillation Onset and Amplitude Across {npos} Positions', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
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
    save_mach_envelope_data(ifn, envelope_save_path)
    
    # =========================================================================
    # Option 2: Load previously processed envelopes and visualize
    # =========================================================================
    # result = load_mach_envelope_data(envelope_save_path)
    # plot_mach_heatmap(result['heatmap_Vx'], result['heatmap_Vy'], 
    #                   result['tarr_downsampled'], result['npos'],
    #                   title_suffix=f"(Run {run_num})")