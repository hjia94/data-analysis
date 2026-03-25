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

import read_hdf5 as rh
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
        pos_array, xpos, ypos, zpos, npos, nshot = rh.read_bmotion_probe_motion(f)

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

def plot_result(Vp_arr, Te_arr, ne_arr, xpos, ypos, t_ls):
    extent = min(xpos), max(xpos), min(ypos), max(ypos)

    sweep_idx = 24
    grid_shape = (61, 61) 
    # Extract the data for the requested sweep and reshape to 2D
    Vp_2d = Vp_arr[:, sweep_idx].reshape(grid_shape)
    Te_2d = Te_arr[:, sweep_idx].reshape(grid_shape)
    ne_2d = ne_arr[:, sweep_idx].reshape(grid_shape)

    # Create a base colormap and tell it to render NaNs as white
    cmap = copy.copy(plt.cm.rainbow)
    cmap.set_bad(color='white')
    interp = 'gaussian'

    # Set up a 1x3 panel figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    t_wt_bias = t_ls[sweep_idx]*1e3 + 14 - 15.5
    fig.suptitle(f'2D Spatial Plasma Profile (t = {t_wt_bias:.2f} ms)', fontsize=32, fontweight='bold')

    # --- 1. Plasma Potential (Vp) ---
    im0 = axs[0].imshow(Vp_2d, origin='lower', cmap=cmap, extent=extent, interpolation=interp, vmin=0, vmax=30)
    axs[0].set_title('Plasma Potential ($V_p$) [V]')
    axs[0].set_xlabel('X Position')
    axs[0].set_ylabel('Y Position')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # --- 2. Electron Temperature (Te) ---
    im1 = axs[1].imshow(Te_2d, origin='lower', cmap=cmap, extent=extent, interpolation=interp, vmin=0, vmax=3)
    axs[1].set_title('Electron Temp ($T_e$) [eV]')
    axs[1].set_xlabel('X Position')
    # Y-label hidden for the middle plot to save space
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # --- 3. Electron Density (ne) ---
    im2 = axs[2].imshow(ne_2d, origin='lower', cmap=cmap, extent=extent, interpolation=interp, vmin=5e11, vmax=5e12)
    axs[2].set_title('Electron Density ($n_e$) [cm$^{-3}$]')
    axs[2].set_xlabel('X Position')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

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
    
    IV_save_path = os.path.join(data_dir, f"{run_num}-sweep-data.npz")
    data = np.load(IV_save_path)
    voltage_data = data['Vswp_arr_rs']
    current_data = data['IswpL_arr_rs']

    save_path = os.path.join(data_dir, f"{run_num}-plasma-data.npz")
    Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err = process_and_save(voltage_data, current_data, save_path)

    # save_IV_data(ifn, IV_save_path)
