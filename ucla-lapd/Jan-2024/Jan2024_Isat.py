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

def get_Isat_ratio(f, adc, npos, nshot, R=9.7, bg_tind=60000):
    '''
    npos    -- number of positions
    nshot   -- number of shots per position
    R       -- resistor value for Isat
    bg_tind -- time index for background subtraction
    '''
    Isat_UL_dic = {}
    Isat_UR_dic = {}
    I_ratio = {}

    for bd in [1,2]:

        control = [('6K Compumotor', bd)]
        if bd == 1:
            st_ind = 0
        if bd == 2:
            st_ind = npos*nshot
            
        data, tarr = rh.read_data(f, bd, 1, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=control)
        Isat_UL = data['signal'].reshape((npos, nshot, -1)) / R
        Isat_UL = Isat_UL - np.mean(Isat_UL[:,:,bg_tind:], axis=-1, keepdims=True) # subtract background
        Isat_UL = gaussian_filter1d(Isat_UL, 25, axis=-1)

        data, tarr = rh.read_data(f, bd, 2, index_arr=slice(st_ind,st_ind+npos*nshot), adc=adc, control=control)
        Isat_UR = data['signal'].reshape((npos, nshot, -1)) / R
        Isat_UR = Isat_UR - np.mean(Isat_UR[:,:,bg_tind:], axis=-1, keepdims=True) # subtract background
        Isat_UR = gaussian_filter1d(Isat_UR, 25, axis=-1)

        I_ratio[bd] = np.mean(Isat_UL/Isat_UR, axis=1)
        Isat_UL_dic[bd] = Isat_UL
        Isat_UR_dic[bd] = Isat_UR

    return tarr, Isat_UL_dic, Isat_UR_dic, I_ratio