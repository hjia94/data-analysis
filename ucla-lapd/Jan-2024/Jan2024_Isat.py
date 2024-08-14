import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

import math
import numpy as np
import h5py
from bapsflib import lapd
import matplotlib.pyplot as plt

import read_hdf5 as rh
from lp_analysis import analyze_IV

from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import find_peaks

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

    return adc, xpos, ypos, zpos, npos, nshot, int_arr, int_tarr


def get_Isat_ratio(f, adc, npos, nshot, area, R=[10,10], bg_tind=60000):
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

    for bd in [1,2]:

        control = [('6K Compumotor', bd)]
        if bd == 1:
            st_ind = 0
            A = area['M3']
        if bd == 2:
            st_ind = npos*nshot
            A = area['M1']

        data, tarr = rh.read_data(f, bd, 1, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=control)
        Isat_UL = data['signal'].reshape((npos, nshot, -1)) / (R[bd-1] * A[0])
        Isat_UL = Isat_UL - np.mean(Isat_UL[:,:,bg_tind:], axis=-1, keepdims=True) # subtract background
        Isat_UL = gaussian_filter1d(Isat_UL, 25, axis=-1)

        data, tarr = rh.read_data(f, bd, 2, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=control)
        Isat_UR = data['signal'].reshape((npos, nshot, -1)) / (R[bd-1] * A[1])
        Isat_UR = Isat_UR - np.mean(Isat_UR[:,:,bg_tind:], axis=-1, keepdims=True) # subtract background
        Isat_UR = gaussian_filter1d(Isat_UR, 25, axis=-1)

        I_ratio[bd] = np.mean(Isat_UL/Isat_UR, axis=1)
        Isat_UL_dic[bd] = Isat_UL
        Isat_UR_dic[bd] = Isat_UR

    return tarr, Isat_UL_dic, Isat_UR_dic, I_ratio
#===============================================================================================================================================
def get_IV(f, adc, npos, nshot, area, cal_fac = [1, 1]):

    Isweep_UL_dic = {}
    Isweep_UR_dic = {}
    

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
        Isweep_UL_dic[bd] = data['signal'].reshape((npos, nshot, -1)) * cal_fac[bd-1] / A[2]
        data, tarr = rh.read_data(f, bd, 4, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=[('6K Compumotor', 1)])
        Isweep_UR_dic[bd] = data['signal'].reshape((npos, nshot, -1)) * cal_fac[bd-1] / A[3]

    return tarr, Vsweep, Isweep_UL_dic, Isweep_UR_dic

def find_IV_tndx(V):
    dV = np.gradient(V)**2
    peaks, _ = find_peaks(dV, prominence=100, distance=1000)
    start_t_ls = []
    stop_t_ls = []
    for p in peaks:
        start_t_ls.append(p-1500)
        stop_t_ls.append(p-100)
    return start_t_ls, stop_t_ls

def find_Te(Vswp, Iswp, plot=False):
    start_t_ls, stop_t_ls = find_IV_tndx(Vswp)
    Te_ls = []
    for i in range(len(start_t_ls)):
        start = start_t_ls[i]
        stop = stop_t_ls[i]
        Vp, Te, ne = analyze_IV(Vswp[start:stop], -Iswp[start:stop], plot=plot)
        Te_ls.append(Te)
    return Te_ls

def get_Te(Vsweep, Isweep, position_range, nshot):

    # Preallocate arrays for Te values
    Te_dic = np.zeros((4, len(position_range), nshot))  # Assuming nshot and positions in the range

    for pos_idx, pos in enumerate(position_range):
        Vswp = Vsweep[pos]
        for j in range(nshot):
            Iswp = Isweep[pos, j]
            Te_dic[:, pos_idx, j] = find_Te(Vswp, Iswp)

    # Calculate mean and standard deviation along the shots axis
    Te_avg = np.mean(Te_dic, axis=2)
    Te_err = np.std(Te_dic, axis=2)

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