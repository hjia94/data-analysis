import math
import numpy as np
import h5py

from bapsflib import lapd
import matplotlib.pyplot as plt

#===============================================================================================================================================

def show_info(f):
    f.overview.print()

def read_probe_motion(f):

    pr_ls = list(f.controls['6K Compumotor'].configs)
    
    # Check number of probes
    if len(pr_ls) == 0:
        print('No probe found.')
    elif len(pr_ls) == 1:
        print('One probe found.')
    elif len(pr_ls) == 2:
        print('2 probes found.')
    else:
        print('More than 2 probes found. Check code before proceeding.')

    pos_array = {}
    for pr_number in pr_ls:
        # Read probe information
        pr = f.controls['6K Compumotor'].configs[pr_number]
        print(pr['probe']['probe type'], pr['probe']['probe name'],' at port ', pr['probe']['port'])

        motion_ls = list(pr['motion lists'].values())
        if len(motion_ls) == 1:
            motion = motion_ls[0]
        else:
            print('More than one motion list found. Check code before proceeding.')

        if len(pr['motion lists']) == 1:
            print('Only one motion list found:', list(pr['motion lists'].keys())[0])
            
        npos = motion['data motion count']
        print('Number of positions:', npos)

        pos_array[pr_number] = f.read_controls([('6K Compumotor', pr_number)])['xyz'] # contains position for each shot

        # Check number of shots per position by looking at how many times each position is repeated
        unique_elements, counts = np.unique(pos_array[pr_number], return_counts=True)
        nshot = np.argmax(np.bincount(counts)) # number of shot per position
        print('Number of shots per position:', nshot)

        # If two or more probes are used; typically one probe moves first and the other moves after the first probe finishes taking data
        # TODO: Only tested with the case of two probes using same motion list
        this_pr_first = not np.all(pos_array[pr_number][0] == pos_array[pr_number][nshot])
        if this_pr_first:
            st_count = 0
            print('This probe moves first in the data run sequence')
        else:
            st_count = counts[-2]
            print('This probe moves after previous probe finish taking data')
        
        # pos_array = pos_array[st_count:st_count+nshot*npos] # only keep moving positions
        
        #====Set up xpos/ypos/zpos arrays using motion list information====
        nx = int(motion['npoints'][0])
        ny = int(motion['npoints'][1])
        nz = int(motion['npoints'][2])

        dx = round(motion['delta'][0], 2)
        dy = round(motion['delta'][1], 2)
        dz = round(motion['delta'][2], 2)

        x0 = round(motion['center'][0], 2)
        y0 = round(motion['center'][1], 2)
        z0 = round(motion['center'][2] + pr['probe']['z'], 2)

        xpos = np.linspace(x0 - (nx-1)*dx/2, x0 + (nx-1)*dx/2, nx).astype(float)
        ypos = np.linspace(y0 - (ny-1)*dy/2, y0 + (ny-1)*dy/2, ny).astype(float)
        zpos = np.linspace(z0 - (nz-1)*dz/2, z0 + (nz-1)*dz/2, nz).astype(float)

    return pos_array, xpos, ypos, zpos, npos, nshot

def read_digitizer_config(f):
    
    grp = f['/Raw data + config/SIS crate']

    for k in grp.keys():
        if 'siscf' in k:
            grp = grp[k]
            print("SIS Crate activated:")
            break
        else:
            print('SIS configuration not found?')

    attributes = grp.attrs
    for attr_name in attributes:
        if 'board types' in attr_name:
            bt_ls = attributes[attr_name]
        if 'config indices' in attr_name:
            ci_ls = attributes[attr_name]

    digi_dict = {}
    for k in grp.keys():
        if 'configurations' in k:
            if '3302' in k:
                adc = 'SIS 3302'
                board_num = int(k.split('configurations[')[1].split(']')[0]) + 1
                print('3302 board', board_num)
                digi_dict[board_num] = []

                for i in range(1,9):
                    enabled = grp[k].attrs['Enabled '+ str(i)]
                    enabled = enabled.decode('utf-8')
                    if enabled == 'TRUE':
                        digi_dict[board_num].append(i)

                    ch_des = grp[k].attrs['Data type '+ str(i)]
                    print('Channel %i -- active: %s -- description: %s' % (i, enabled, ch_des.decode('utf-8')))

    return adc, digi_dict


def read_data(f, board_num, chan_num, index_arr=None, adc='SIS 3302', control=None):

    digi_ls = list(f.digitizers)
    if len(digi_ls) == 1:
        digitizer = digi_ls[0]
    else:
        print('More than one digitizer found. The first one has been selected.')

    config_ls = f.digitizers[digitizer].active_configs
    if len(config_ls) == 1:
        config_name = config_ls[0]
    else:
        print('More than one configuration found. The first one has been selected.')
    
    if index_arr == None:
        data = f.read_data(board_num,chan_num, add_controls=control, digitizer=digitizer, adc=adc, config_name=config_name)
    else:
        data = f.read_data(board_num,chan_num, index=index_arr, add_controls=control, digitizer=digitizer, adc=adc, config_name=config_name)

    nt = data['signal'].shape[1]
    tarr = np.arange(nt) * data.dt.value
    
    return data, tarr

#===============================================================================================================================================
def unpack_datarun_sequence(f):

    sequence_list = f['Raw data + config/Data run sequence/Data run sequence']
    message_array = np.array([])
    status_array = np.array([])
    timestamp_array = np.array([])

    for i in range(len(sequence_list)):
        output = sequence_list[i]

        # Extract elements
        message = output[0].decode('utf-8')  # Convert bytes to string
        message_array = np.append(message_array, message)

        # Don't know what output[1] is
        # output[2] seems to be an index

        status = output[3].decode('utf-8')  # Convert bytes to string
        status_array = np.append(status_array, status)

        timestamp = output[4]
        timestamp_array = np.append(timestamp_array, timestamp)

    return message_array, status_array, timestamp_array
#===============================================================================================================================================
def read_magnetic_field(f):

    mdata = f.read_msi('Magnetic field')
    Bdata = mdata['magnetic field']

    port_ls = np.linspace(60, -6, len(Bdata[0]))

    return Bdata[0], port_ls

def read_interferometer_old(f):
    '''
    TODO: need to fix returned density factor
    '''
    int_data = f.read_msi('Interferometer array')
    int_dic = int_data.info
    int_arr = int_data['signal'][:,1]
    den_factor = int_dic['n_bar_L'][1] * 1e-2

    t0 = int_dic['t0'][1]
    dt = int_dic['dt'][1]

    nt = int_arr.shape[1]
    int_tarr = np.linspace(t0, t0+dt*nt, nt)

    return int_arr, int_tarr, den_factor
#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':

    ifn = r"C:\data\LAPD\JAN2024_diverging_B\02_2cmXYline_M1P24_M3P27_2024-01-26_15.16.39.hdf5"
    f = lapd.File(ifn)
    f.overview.print()