from multiprocessing import process
import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

import time
import datetime
import os
import copy
import numpy as np
import h5py
from bapsflib import lapd
import matplotlib.pyplot as plt

import read_hdf5_bapsflib as rh
from lp_analysis import find_sweep_indices, reshape_IV
from lp_iv_analysis import analyze_IV_safe

from scipy.ndimage import gaussian_filter1d

Aprobe = 2e-3 # 2X1mm probe, in cm^2
cal_fac = [1, 1] # Ratio between L and R side of probe
#===============================================================================================================================================

def get_IV_arr(f, adc, npos, nshot):
    
    '''
    Board 4:
    Chan4:LP I, p32 L 
    Chan5:LP V, p32 L
    Chan6:LP I, p32 R
    Chan7:LP V, p32 R
    '''
    data, tarr = rh.read_data(f, 4, 5, index_arr=slice(0,npos*nshot), adc=adc)
    Vsweep = data['signal'].reshape((npos, nshot, -1)) * 100 # X50 probe
    Vsweep = np.mean(Vsweep, axis=1)

    data, tarr = rh.read_data(f, 4, 4, index_arr=slice(npos*nshot), adc=adc)
    IsweepL_arr = data['signal'].reshape((npos, nshot, -1)) / (7.2 * Aprobe) # number is resistor value
    data, tarr = rh.read_data(f, 4, 6, index_arr=slice(npos*nshot), adc=adc)
    IsweepR_arr = data['signal'].reshape((npos, nshot, -1)) / (25 * Aprobe) # number is resistor value

    return tarr, Vsweep, IsweepL_arr, IsweepR_arr

    # Use the following to check if sweep timing and indices are correct
    # plt.figure()
    # plt.plot(tarr, Vswp_arr[0], label='Voltage')
    # for st, sp in zip(start_t_ls, stop_t_ls):
    #     plt.axvline(tarr[st], color='r', linestyle='--')
    #     plt.axvline(tarr[sp], color='r', linestyle='--')

def save_IV_data(ifn, save_path):

    with lapd.File(ifn) as f:
        
        adc, digi_dict = rh.read_digitizer_config(f)
        # read probe motion into arrays
        pos_array, xpos, ypos, zpos, npos, nshot = rh.read_probe_motion_bmotion(f)

        # read probe signal into arrays
        tarr, Vswp_arr, IswpL_arr, IswpR_arr = get_IV_arr(f, adc, npos, nshot)

        # Reshape arrays to include swept traces only
        start_t_ls, stop_t_ls = find_sweep_indices(Vswp_arr[0], padding=10)

        # Calculate middle indices for each sweep and extract corresponding times from tarr
        mid_indices = [(start + stop) // 2 for start, stop in zip(start_t_ls, stop_t_ls)]
        data_timestamp = tarr[mid_indices]
        print(f"Number of sweeps: {len(data_timestamp)}")

        # Reshape the voltage and current arrays to only include the sweep data
        Vswp_arr_rs, IswpL_arr_rs = reshape_IV(Vswp_arr, IswpL_arr, start_t_ls, stop_t_ls, 5)
        Vswp_arr_rs, IswpR_arr_rs = reshape_IV(Vswp_arr, IswpR_arr, start_t_ls, stop_t_ls, 5)

        # Apply smoothing to the current arrays
        print("Applying smoothing to current arrays...")
        IswpL_arr_rs = gaussian_filter1d(IswpL_arr_rs, 10, axis=-1)
        IswpR_arr_rs = gaussian_filter1d(IswpR_arr_rs, 10, axis=-1)    

    # Save everything into a .npz file for later analysis
    np.savez(save_path, Vswp_arr_rs=Vswp_arr_rs, IswpL_arr_rs=IswpL_arr_rs, IswpR_arr_rs=IswpR_arr_rs,
             data_timestamp=data_timestamp, xpos=xpos, ypos=ypos, npos=npos, nshot=nshot)
    print(f"Saved to: {save_path}")

def process_and_save(voltage_data, current_data, save_path):
    """
    Loops through the multi-dimensional Langmuir probe dataset, extracting 
    plasma parameters. Averages the valid shots for each location/sweep combination,
    calculates the standard error, and outputs 2D arrays.
    Saves progress incrementally to prevent data loss.
    """
    n_locs, n_shots, n_sweeps, _ = current_data.shape
    
    # Pre-allocate 2D output arrays with NaNs (locs, sweeps)
    Vp_arr = np.full((n_locs, n_sweeps), np.nan)
    Te_arr = np.full((n_locs, n_sweeps), np.nan)
    ne_arr = np.full((n_locs, n_sweeps), np.nan)
    
    # Pre-allocate 2D error arrays for the error bars
    Vp_err = np.full((n_locs, n_sweeps), np.nan)
    Te_err = np.full((n_locs, n_sweeps), np.nan)
    ne_err = np.full((n_locs, n_sweeps), np.nan)
    
    total_traces = n_locs * n_shots * n_sweeps
    print(f"Starting batch processing of {total_traces} traces across {n_locs} locations...")
    print(f"Averaging {n_shots} shots per sweep...")
    
    start_time = time.time()
    traces_completed = 0
    fail_count = 0  
    
    for loc in range(n_locs):
        for swp in range(n_sweeps):
            # The voltage trace applies to all shots at this location/sweep
            V_trace = voltage_data[loc, swp, :]
            
            # Temporary storage for the shots in this specific sweep
            temp_Vp = []
            temp_Te = []
            temp_ne = []
            
            for sht in range(n_shots):
                I_trace = current_data[loc, sht, swp, :]
                trace_id = f"Loc:{loc}|Shot:{sht}|Swp:{swp}"
                
                # Analyze trace
                Vp, Te, ne = analyze_IV_safe(V_trace, I_trace, file_name=trace_id)
                
                # Track failures on a per-trace basis
                if np.isnan(Vp):
                    fail_count += 1
                else:
                    # Only keep valid numbers for averaging
                    temp_Vp.append(Vp)
                    temp_Te.append(Te)
                    temp_ne.append(ne)
                    
                traces_completed += 1
                
            # --- AVERAGING LOGIC ---
            # Calculate the mean and standard error of the mean (SEM) for valid shots
            
            # Vp
            if len(temp_Vp) > 0:
                Vp_arr[loc, swp] = np.mean(temp_Vp)
                # If more than 1 valid shot, calculate error: std_dev / sqrt(N)
                Vp_err[loc, swp] = np.std(temp_Vp, ddof=1) / np.sqrt(len(temp_Vp)) if len(temp_Vp) > 1 else np.nan
                
            # Te
            if len(temp_Te) > 0:
                Te_arr[loc, swp] = np.mean(temp_Te)
                Te_err[loc, swp] = np.std(temp_Te, ddof=1) / np.sqrt(len(temp_Te)) if len(temp_Te) > 1 else np.nan
                
            # ne
            # Because ne can be NaN if Te was forced to 0, we need to filter NaNs out of temp_ne specifically
            valid_ne = [n for n in temp_ne if not np.isnan(n)]
            if len(valid_ne) > 0:
                ne_arr[loc, swp] = np.mean(valid_ne)
                ne_err[loc, swp] = np.std(valid_ne, ddof=1) / np.sqrt(len(valid_ne)) if len(valid_ne) > 1 else np.nan

        # ==========================================
        # INCREMENTAL SAVE
        # ==========================================
        # Save all 6 arrays to the file
        np.savez(save_path, 
                 Vp_arr=Vp_arr, Te_arr=Te_arr, ne_arr=ne_arr,
                 Vp_err=Vp_err, Te_err=Te_err, ne_err=ne_err)
        
        # Print progress, ETA, and fail stats every 10 locations
        if (loc + 1) % 10 == 0 or (loc + 1) == n_locs:
            elapsed = time.time() - start_time
            time_per_trace = elapsed / traces_completed
            traces_left = total_traces - traces_completed
            eta_seconds = traces_left * time_per_trace
            
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            percent_done = (traces_completed / total_traces) * 100
            
            current_fail_rate = (fail_count / traces_completed) * 100
            
            print(f"Progress: {percent_done:5.2f}% (Loc: {loc+1}/{n_locs}) | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                  f"Speed: {time_per_trace*1000:.1f} ms/trace | "
                  f"Fails: {fail_count} ({current_fail_rate:.1f}%)")
            
    # ==========================================
    # FINAL SUMMARY
    # ==========================================            
    total_time = time.time() - start_time
    final_time_str = str(datetime.timedelta(seconds=int(total_time)))
    final_fail_rate = (fail_count / total_traces) * 100
    
    print("\n" + "="*55)
    print("BATCH PROCESSING COMPLETE")
    print("="*55)
    print(f"Total Time:    {final_time_str}")
    print(f"Total Traces:  {total_traces}")
    print(f"Total Fails:   {fail_count}")
    print(f"Fail Rate:     {final_fail_rate:.2f}%")
    print(f"Data saved to: {save_path}")
    print("="*55)
    
    return Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err


def load_data(data_dir, run_num):
    save_path = os.path.join(data_dir, f"{run_num}-sweep-data.npz")
    data = np.load(save_path)
    t_ls = data['data_timestamp']

    ps_path = os.path.join(data_dir, f"{run_num}-plasma-data.npz")
    ps_data = np.load(ps_path)
    Vp_arr = ps_data['Vp_arr']
    Te_arr = ps_data['Te_arr']
    ne_arr = ps_data['ne_arr']
    Vp_err = ps_data['Vp_err']
    Te_err = ps_data['Te_err']
    ne_err = ps_data['ne_err']

    return Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls

def plot_result(ne_arr, xpos, ypos, t_ls):
    """
    Plot electron density for all time frames in a grid layout.
    
    Parameters
    ----------
    ne_arr : np.ndarray
        Electron density array (n_locs, n_sweeps)
    xpos : np.ndarray
        X positions of probe locations [cm]
    ypos : np.ndarray
        Y positions of probe locations [cm]
    t_ls : np.ndarray
        Time array for sweeps [s]
    """
    extent = (min(xpos), max(xpos), min(ypos), max(ypos))
    
    n_locs, n_sweeps = ne_arr.shape
    grid_shape = (len(ypos), len(xpos))
    
    # Create a base colormap and tell it to render NaNs as white
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad(color='white')
    interp = 'gaussian'
    
    # Get global ne range for consistent colormap
    ne_valid_all = ne_arr[~np.isnan(ne_arr)]
    if len(ne_valid_all) > 0:
        ne_vmin = ne_valid_all.min()
        ne_vmax = ne_valid_all.max()
    else:
        ne_vmin, ne_vmax = 0, 1  # Default range if no valid data
    
    # Determine subplot layout (5 columns)
    ncols = 5
    nrows = (n_sweeps + ncols - 1) // ncols
    
    # Create figure
    fig, axs = plt.subplots(nrows, ncols, figsize=(18, 4*nrows), squeeze=False)
    axs = axs.flatten()
    
    im_obj = None  # To store image object for colorbar
    
    # Plot each sweep
    for sweep_idx in range(n_sweeps):
        ne_2d = ne_arr[:, sweep_idx].reshape(grid_shape)
        t_sweep = t_ls[sweep_idx] * 1e3  # Convert to ms
        
        ax = axs[sweep_idx]
        im_obj = ax.imshow(ne_2d, origin='lower', cmap=cmap, extent=extent, 
                           interpolation=interp, vmin=ne_vmin, vmax=ne_vmax)
        
        ax.set_title(f'{t_sweep:.1f} ms', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_sweeps, len(axs)):
        axs[idx].axis('off')
    
    # Single colorbar at bottom
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.12,
                        wspace=0.15, hspace=0.4)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.70, 0.02])
    fig.colorbar(im_obj, cax=cbar_ax, orientation='horizontal',
                label='Electron Density $n_e$ [cm$^{-3}$]')

    plt.show()

def plot_result_line(Vp_arr, Te_arr, ne_arr, xpos, ypos, t_ls, tndx_list):
    """
    Plot center line across y=0 showing Vp, Te, and ne at selected time indices.
    
    Parameters
    ----------
    Vp_arr : np.ndarray
        Plasma potential array (n_locs, n_sweeps)
    Te_arr : np.ndarray
        Electron temperature array (n_locs, n_sweeps)
    ne_arr : np.ndarray
        Electron density array (n_locs, n_sweeps)
    xpos : np.ndarray
        X positions of probe locations [cm]
    ypos : np.ndarray
        Y positions of probe locations [cm]
    t_ls : np.ndarray
        Time array for sweeps [s]
    tndx_list : list
        List of time indices to plot
    """
    # Find the index of the center line (closest to y=0)
    y_center_idx = np.argmin(np.abs(ypos))
    
    nx = len(xpos)
    ny = len(ypos)
    
    # Calculate linear indices for the center line
    line_indices = np.arange(ny) * nx + y_center_idx
    
    # Create figure with 3 subplots for Vp, Te, ne
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # Color map for different time indices
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tndx_list)))
    
    # Iterate through selected time indices
    for color, t_idx in zip(colors, tndx_list):
        # Ensure time index is valid
        if t_idx >= Vp_arr.shape[1]:
            print(f"Warning: time index {t_idx} exceeds array size {Vp_arr.shape[1]}")
            continue
        
        t_val = t_ls[t_idx] * 1e3  # Convert to ms
        
        # Extract center line data
        Vp_line = Vp_arr[line_indices, t_idx]
        Te_line = Te_arr[line_indices, t_idx]
        ne_line = ne_arr[line_indices, t_idx]
        
        # Get corresponding x positions
        x_line = xpos
        
        # Plot Vp
        axs[0].plot(x_line, Vp_line, 'o-', color=color, 
                   label=f't = {t_val:.2f} ms', linewidth=2, markersize=5)
        
        # Plot Te
        axs[1].plot(x_line, Te_line, 's-', color=color, 
                   label=f't = {t_val:.2f} ms', linewidth=2, markersize=5)
        
        # Plot ne
        axs[2].plot(x_line, ne_line, '^-', color=color, 
                   label=f't = {t_val:.2f} ms', linewidth=2, markersize=5)
    
    # Set labels and titles
    axs[0].set_title(f'Plasma Parameter (Center Line at y ≈ {ypos[y_center_idx]:.2f} cm)', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Vp [V]', fontsize=11)
    axs[0].legend(fontsize=9, loc='best')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].set_ylabel('Te [eV]', fontsize=11)
    axs[1].legend(fontsize=9, loc='best')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].set_xlabel('X Position [cm]', fontsize=11)
    axs[2].set_ylabel('ne [cm$^{-3}$]', fontsize=11)
    axs[2].legend(fontsize=9, loc='best')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':

    ifn = r"D:\data\LAPD\Mar26-data\11-lp-sweep-xyplane-bias.hdf5"
    # Extract directory and run number from ifn
    data_dir = os.path.dirname(ifn)
    run_num = os.path.basename(ifn).split('-')[0]
    
    # Load data
    Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls = load_data(data_dir, run_num)
    
    # Plot result
    with lapd.File(ifn) as f:
        pos_dict, xpos, ypos, zpos, npos, nshot = rh.read_probe_motion_bmotion(f)

    # plot_result(ne_arr, xpos, ypos, t_ls)
    tndx_list = [0, 7, 15, 20]
    plot_result_line(Vp_arr, Te_arr, ne_arr, xpos, ypos, t_ls, tndx_list)
