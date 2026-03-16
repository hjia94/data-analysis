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

def get_IV(f, adc, npos, nshot, area, cal_fac = [1, 1]):
    
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
    IsweepL_arr = data['signal'].reshape((npos, nshot, -1))
    data, tarr = rh.read_data(f, 4, 6, index_arr=slice(npos*nshot), adc=adc)
    IsweepR_arr = data['signal'].reshape((npos, nshot, -1))

    return tarr, Vsweep, IsweepL_arr, IsweepR_arr

