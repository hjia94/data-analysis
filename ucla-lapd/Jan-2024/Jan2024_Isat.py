import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

import math
import numpy as np
import h5py
from bapsflib import lapd
import matplotlib.pyplot as plt

import read_hdf5 as rh
from lp_analysis import analyze_IV, derivative

from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import find_peaks, savgol_filter

#===============================================================================================================================================
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#===============================================================================================================================================
def init_read(ifn):
    with lapd.File(ifn) as f:
        adc, digi_dict = rh.read_digitizer_config(f)
        pos_array, xpos, ypos, zpos, npos, nshot = rh.read_probe_motion(f)
        Bdata, port_ls = rh.read_magnetic_field(f)
        int_arr, int_tarr, d_phi = rh.read_interferometer_old(f)

        plt.figure(figsize=(10, 5))
        plt.plot(port_ls, Bdata)
        plt.title(ifn[ifn.index('\\0'):])
        plt.ylabel('B (G)')
        plt.xlabel('Port #')

        plt.axvline(x=24, color='r', linestyle='--', label='LP P24')
        plt.axvline(x=27, color='g', linestyle='--', label='LP P27')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.xlabel('time (ms)')
        plt.ylabel('Estimated n_e (cm^-3)')
        plt.plot(int_tarr*1e3, int_arr[0]*d_phi, label='start of datarun')
        plt.plot(int_tarr*1e3, int_arr[1]*d_phi, label='end of datarun')
        plt.legend()
        plt.tight_layout()

    return adc, digi_dict, pos_array, xpos, ypos, zpos, npos, nshot, int_arr, int_tarr


def get_Isat_ratio(f, adc, digi_dict, npos, nshot, area, chL=1, chR=2, R=[10,10], bg_tind=60000):
    '''
    Inputs:
    npos    -- number of positions
    nshot   -- number of shots per position
    R       -- resistor value for Isat
    bg_tind -- time index for background subtraction
    
    Outputs:
    tarr -- second
    Isat_UL_dic, Isat_UR_dic -- A/area
    I_ratio: Isat_UL/Isat_UR (facing south/north)
    '''
    Isat_UL_dic = {}
    Isat_UR_dic = {}
    I_ratio = {}

    for bd in digi_dict.keys():

        control = [('6K Compumotor', bd)]
        if bd == 1:
            st_ind = 0
            A = area['M3']
        if bd == 2:
            st_ind = npos*nshot
            A = area['M1']

        data, tarr = rh.read_data(f, bd, chL, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=control)
        Isat_UL = data['signal'].reshape((npos, nshot, -1)) / (R[bd-1] * A[0])
        Isat_UL = Isat_UL - np.mean(Isat_UL[:,:,bg_tind:], axis=-1, keepdims=True) # subtract background
        Isat_UL = gaussian_filter1d(Isat_UL, 25, axis=-1)

        data, tarr = rh.read_data(f, bd, chR, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=control)
        Isat_UR = data['signal'].reshape((npos, nshot, -1)) / (R[bd-1] * A[1])
        Isat_UR = Isat_UR - np.mean(Isat_UR[:,:,bg_tind:], axis=-1, keepdims=True) # subtract background
        Isat_UR = gaussian_filter1d(Isat_UR, 25, axis=-1)

        I_ratio[bd] = np.mean((Isat_UL-Isat_UR)/(Isat_UL+Isat_UR), axis=1)
        Isat_UL_dic[bd] = Isat_UL
        Isat_UR_dic[bd] = Isat_UR

    return tarr, Isat_UL_dic, Isat_UR_dic, I_ratio
#===============================================================================================================================================
def get_IV(f, adc, npos, nshot, area, cal_fac = [1, 1]):

    IsweepL_dic = {}
    IsweepR_dic = {}
    
    data, tarr = rh.read_data(f, 1, 7, index_arr=slice(0,npos*nshot), adc=adc, control=[('6K Compumotor', 1)])
    Vsweep = data['signal'].reshape((npos, nshot, -1)) * 100 # X50 probe
    Vsweep = np.mean(Vsweep, axis=1)

    for bd in [1,2]:
        if bd == 1:
            st_ind = 0
            A = area['M3']
        if bd == 2:
            st_ind = npos*nshot
            A = area['M1']

        data, tarr = rh.read_data(f, bd, 3, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=[('6K Compumotor', 1)])
        IsweepL_dic[bd] = data['signal'].reshape((npos, nshot, -1)) * cal_fac[bd-1] / A[2]
        data, tarr = rh.read_data(f, bd, 4, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=[('6K Compumotor', 1)])
        IsweepR_dic[bd] = data['signal'].reshape((npos, nshot, -1)) * cal_fac[bd-1] / A[3]

    return tarr, Vsweep, IsweepL_dic, IsweepR_dic
#--------------------------------------------------------------
def find_IV_tndx(V):
    dV = np.gradient(V)**2
    peaks, _ = find_peaks(dV, prominence=100, distance=1000)
    start_t_ls = []
    stop_t_ls = []
    for p in peaks:
        start_t_ls.append(p-1500)
        stop_t_ls.append(p-100)
    return start_t_ls, stop_t_ls
#--------------------------------------------------------------
def reshape_IV(Vsweep_arr, Isweep_arr):

    start_t_ls, stop_t_ls = find_IV_tndx(Vsweep_arr[0])
    # Initialize an empty list to store the chunks
    I_chunks = []
    V_chunks = []

    # Loop through the pairs of start_t_ls and stop_t_ls
    for start, stop in zip(start_t_ls, stop_t_ls):

        I_chunks.append(Isweep_arr[:, :, start:stop])
        V_chunks.append(Vsweep_arr[:, start:stop])

    # Stack the list of chunks into a new array
    Isweep_reshaped = np.stack(I_chunks, axis=2)
    Vsweep_reshaped = np.stack(V_chunks, axis=1)

    return Vsweep_reshaped, -Isweep_reshaped

def smooth_Isweep(IsweepL_dic, IsweepR_dic, w=20, p=1):
    ss_L_dic = {}
    ss_R_dic = {}
    for bd in [1,2]:
        ss_L_dic[bd] = savgol_filter(IsweepL_dic[bd], window_length=w, polyorder=p, axis=-1)
        ss_R_dic[bd] = savgol_filter(IsweepR_dic[bd], window_length=w, polyorder=p, axis=-1)
    return ss_L_dic, ss_R_dic
#--------------------------------------------------------------
def get_dIdV(Iswp_arr, Vswp_arr): # Work in progress

    dIdV_arr = np.zeros_like(Iswp_arr)
    max_inds_arr = np.zeros(Iswp_arr.shape[0:3], dtype=int)
    npos, nswp, ntime = Vswp_arr.shape

    for pos in range(npos):    
        for i in range(nswp):
            dIdV, max_inds = derivative(Iswp_arr[pos, :, i], Vswp_arr[pos, i])
            dIdV_arr[pos,:,i] = dIdV
            max_inds_arr[pos,:,i] = max_inds

    return dIdV_arr, max_inds_arr
#--------------------------------------------------------------
def find_Te(Vswp, Iswp, plot=False): # TODO: adjust with reshape_IV
    start_t_ls, stop_t_ls = find_IV_tndx(Vswp)
    Te_ls = []
    for i in range(len(start_t_ls)):
        start = start_t_ls[i]
        stop = stop_t_ls[i]
        Vp, Te, ne = analyze_IV(Vswp[start:stop], Iswp[start:stop], plot=plot)
        Te_ls.append(Te)
    return Te_ls

def get_Te_all(Vsweep, Isweep, position_range, nshot): # TODO: adjust with reshape_IV

    # Preallocate arrays for Te values
    Te_arr = np.zeros((4, len(position_range), nshot))  # Assuming nshot and positions in the range

    for pos_idx, pos in enumerate(position_range):
        Vswp = Vsweep[pos]
        for j in range(nshot):
            Iswp = Isweep[pos, j]
            Te_arr[:, pos_idx, j] = find_Te(Vswp, Iswp)

    # Calculate mean and standard deviation along the shots axis
    Te_avg = np.mean(Te_arr, axis=2)
    Te_err = np.std(Te_arr, axis=2)

    return Te_avg, Te_err

#===============================================================================================================================================
def setup_2subplots(xlabel='radial position (cm)', ylabel='Isat (A/cm^2)', title_A='port 27', title_B='port 24', grid=True):

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_title(title_A)
    ax[1].set_title(title_B)
    ax[1].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[1].set_ylabel(ylabel)
    ax[0].grid(grid)
    ax[1].grid(grid)

    return ax

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':
    ifn = r"C:\data\LAPD\JAN2024_diverging_B\04_2cmXYline_M1P24_M3P27_2024-01-26_19.17.54.hdf5"
    pr_area = {
    'M3': np.array([1,1,1,1.15]) * 0.2*0.4,
    'M1': np.array([1,1,1.14,1]) * 0.1*0.4
        } # [UL, UR, LL, LR] ; unit:cm^2
    
    adc, xpos, ypos, zpos, npos, nshot, int_arr, int_tarr = init_read(ifn)

    with lapd.File(ifn) as f:
        tarr, Vsweep, Isweep_UL_dic, Isweep_UR_dic = get_IV(f, adc, npos, nshot, pr_area, cal_fac = [2, 1])

    plt.figure()
    plt.plot(Vsweep[10])
    start_t_ls, stop_t_ls = find_IV_tndx(Vsweep[10])
    for i in range(len(start_t_ls)):
        plt.axvline(start_t_ls[i], color='r')
        plt.axvline(stop_t_ls[i], color='g')