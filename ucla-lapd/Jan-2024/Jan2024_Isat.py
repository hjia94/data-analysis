import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\ucla-lapd")

import math
import numpy as np
import h5py
from bapsflib import lapd
import matplotlib.pyplot as plt

import read_hdf5 as rh

from scipy.ndimage import gaussian_filter1d, gaussian_filter

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