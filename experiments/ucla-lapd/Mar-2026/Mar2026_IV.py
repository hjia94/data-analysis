import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from data_analysis.io import open_lapd
from data_analysis.plasma.langmuir import prepare_sweep_data, load_plasma_data
from data_analysis.utils import run_num_of

Aprobe = 2e-3 # 2X1mm probe, in cm^2
cal_fac = [1, 1] # Ratio between L and R side of probe
#===============================================================================================================================================

def get_IV_arr(sess, adc, npos, nshot):

    '''
    Board 4:
    Chan4:LP I, p32 L
    Chan5:LP V, p32 L
    Chan6:LP I, p32 R
    Chan7:LP V, p32 R
    '''
    data, tarr = sess.read_data(4, 5, index_arr=slice(0,npos*nshot), adc=adc)
    Vsweep = data['signal'].reshape((npos, nshot, -1)) * 100 # X50 probe
    Vsweep = np.mean(Vsweep, axis=1)

    data, tarr = sess.read_data(4, 4, index_arr=slice(npos*nshot), adc=adc)
    IsweepL_arr = data['signal'].reshape((npos, nshot, -1)) / (7.2 * Aprobe) # number is resistor value
    data, tarr = sess.read_data(4, 6, index_arr=slice(npos*nshot), adc=adc)
    IsweepR_arr = data['signal'].reshape((npos, nshot, -1)) / (25 * Aprobe) # number is resistor value

    return tarr, Vsweep, IsweepL_arr, IsweepR_arr

    # Use the following to check if sweep timing and indices are correct
    # plt.figure()
    # plt.plot(tarr, Vswp_arr[0], label='Voltage')
    # for st, sp in zip(start_t_ls, stop_t_ls):
    #     plt.axvline(tarr[st], color='r', linestyle='--')
    #     plt.axvline(tarr[sp], color='r', linestyle='--')

def save_IV_data(ifn, save_path):

    with open_lapd(ifn).session() as sess:

        adc, digi_dict = sess.digitizer_config()
        # read probe motion into arrays
        pos_array, xpos, ypos, zpos, npos, nshot = sess.positions()

        # read probe signal into arrays
        tarr, Vswp_arr, IswpL_arr, IswpR_arr = get_IV_arr(sess, adc, npos, nshot)

        # Sweep detection -> reshape -> smoothing (shared batch pipeline), once
        # per current array; the detected sweeps/timestamps are identical since
        # both use the same voltage trace.
        Vswp_arr_rs, IswpL_arr_rs, data_timestamp, *_ = prepare_sweep_data(
            tarr, Vswp_arr, IswpL_arr, trim_percent=5)
        _, IswpR_arr_rs, *_ = prepare_sweep_data(
            tarr, Vswp_arr, IswpR_arr, trim_percent=5)

    # Save everything into a .npz file for later analysis
    np.savez(save_path, Vswp_arr_rs=Vswp_arr_rs, IswpL_arr_rs=IswpL_arr_rs, IswpR_arr_rs=IswpR_arr_rs,
             data_timestamp=data_timestamp, xpos=xpos, ypos=ypos, npos=npos, nshot=nshot)
    print(f"Saved to: {save_path}")

# Batch analysis (process_iv_and_save) + npz loading (load_plasma_data) live in
# data_analysis.plasma.langmuir now -- call them from there.


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
    run_num = run_num_of(ifn)
    
    # Load data
    Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls = load_plasma_data(data_dir, run_num)
    
    # Plot result
    with open_lapd(ifn).session() as sess:
        pos_dict, xpos, ypos, zpos, npos, nshot = sess.positions()

    # plot_result(ne_arr, xpos, ypos, t_ls)
    tndx_list = [0, 7, 15, 20]
    plot_result_line(Vp_arr, Te_arr, ne_arr, xpos, ypos, t_ls, tndx_list)
