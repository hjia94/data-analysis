import math
import numpy
import h5py

from bapsflib import lapd
import matplotlib.pyplot as plt


def show_info(f):
    f.overview.print()

def read_probe_motion(f, number):

    pr = f.controls['6K Compumotor'].configs[number]

    print(pr['probe']['probe type'], pr['probe']['probe name'])

    motion = list(pr['motion lists'].values())[0]

    nx = motion['npoints'][0]
    ny = motion['npoints'][1]
    nz = motion['npoints'][2]

    # dx = motion['delta'][0]
    # dy = motion['delta'][1]
    # dz = motion['delta'][2]

    # x0 = motion['center'][0]
    # y0 = motion['center'][1]
    # z0 = motion['center'][2] + pr['probe']['z']

    pos_array = f.read_controls([('6K Compumotor', 3)])['xyz']

    npos = motion['data motion count']
    nshot = int(len(pos_array) / npos)

    xpos = pos_array[::nshot][:nx,0]
    ypos = pos_array[::nshot][::ny,1]
    zpos = pr['probe']['z']

    return pos_array, xpos, ypos, zpos, npos, nshot



def read_data(f, board_num, chan_num, index_arr=None, adc='SIS 3302'):

    digitizer = list(f.digitizers)[0]

    config_name = f.digitizers[digitizer].active_configs[0]
    
    if index_arr == None:
        data = f.read_data(board_num,chan_num, add_controls=[('6K Compumotor',3)], digitizer=digitizer, adc=adc, config_name=config_name)
    else:
        data = f.read_data(board_num,chan_num, index=index_arr, add_controls=[('6K Compumotor',3)], digitizer=digitizer, adc=adc, config_name=config_name)

    return data['signal']

def data_time(f, board_num, chan_num, adc='SIS 3302'):

    digitizer = list(f.digitizers)[0]

    config_name = f.digitizers[digitizer].active_configs[0]
        
    data = f.read_data(board_num,chan_num, 0, add_controls=[('6K Compumotor',3)], digitizer=digitizer, adc=adc, config_name=config_name)

    nt = data['signal'].shape[1]
    tarr = numpy.arange(nt) * data.dt
    
    return nt, data.dt, tarr


#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':

    ifn = r"C:\data\LAPD\JAN2024_diverging_B\02_2cmXYline_M1P24_M3P27_2024-01-26_15.16.39.hdf5"
    f = lapd.File(ifn)
    f.overview.print()