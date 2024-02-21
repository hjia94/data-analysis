# -*- coding: utf-8 -*-
"""
File Description:
This file contains the implementation of functions for reading and processing data from an HDF5 file.

File history:
Original version of these functions are written by Pat for reading hdf5 data for process plasma.
Used and modified betweeen 2018-2020 by Jia depending on the data acquisition structure.
This file combines common functions from read_3Ddata.py, read_field_3D.py, read_field_3D_4chan.py, read_field_XZplane.py that were created during 2018-2020.


Functions:
- check_input_file(ifn): Checks if the input file exists.
- save_to_file(sfn, data): Saves data to a file.
- read_save_file(sfn): Reads data from a saved file.
- print_data_objects(file_path): Prints the objects present in an HDF5 file.
- channel_description(chnum, h5obj): Returns the description of a channel in an HDF5 file.
- get_header(file_path, print_info=False): Retrieves the header information from an HDF5 file.
- read_channel_data(file_path, chnum, print_description=False): Reads the data of a specific channel from an HDF5 file.
- read_position_data(file_path): Reads the position data from an HDF5 file.
- generate_time_array(h, data): Generates a time array based on the header information and data length.

Author: Jia Han (with GitHub Copilot)
Created on: 2024-02-20

"""

import h5py
import os
import numpy as np
from LeCroy_Scope_Header import LeCroy_Scope_Header

#======================================================================================
#======================================================================================

def check_input_file(ifn):
    """
    Checks if the input file exists.

    Parameters:
    ifn (str): The path to the input file.

    Raises:
    RuntimeError: If the input file does not exist.
    """
    if not os.path.isfile(ifn):
        raise RuntimeError('input file "', ifn, '" not found', sep='')

def save_to_file(sfn, data):
    """
    Saves data to a file.

    Parameters:
    sfn (str): The path to the output file.
    data (numpy.ndarray): The data to be saved.
    """
    np.save(sfn, data)
    print("Saved to file", sfn)

def read_save_file(sfn):
    """
    Reads data from a saved file.

    Parameters:
    sfn (str): The path to the saved file.

    Returns:
    numpy.ndarray: The data read from the file.
    """
    sfn = os.path.splitext(ifn)[0] + ".cache.npy"
    if os.path.isfile(sfn):
        return np.load(sfn)

#======================================================================================
#======================================================================================

def print_data_objects(file_path):
    """
    Prints the objects present in an HDF5 file.

    Parameters:
    file_path (str): The path to the HDF5 file.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print("Objects in hdf5 file are:")
            for name in f:
                print("   ", name)
    except IOError:
        print(f"Cannot open file {file_path}.")

def channel_description(chnum, h5obj):
    """
    Returns the description of a channel in an HDF5 file.

    Parameters:
    chnum (int): The channel number.
    h5obj (h5py.Dataset): The HDF5 dataset object representing the channel.

    Returns:
    str: The channel description.
    """
    try:
        return "CH%i: "%(chnum) + h5obj.attrs['description'] + "   -> " + str(h5obj.shape)
    except KeyError:
        return "CH%i: "%(chnum) + "  (no description entry)" + "   -> " + str(h5obj.shape)

def get_header(file_path, print_info=False):
    """
    Retrieves the header information from an HDF5 file.

    Parameters:
    file_path (str): The path to the HDF5 file.
    print_info (bool, optional): Whether to print the header information. Defaults to False.

    Returns:
    LeCroy_Scope_Header: The header object.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            channels = f["Acquisition/LeCroy_scope"].keys()
            for channel in channels:
                if "Headers" in f["Acquisition/LeCroy_scope"][channel]:
                    hdr_bytes = f["Acquisition/LeCroy_scope"][channel]["Headers"][1]   # 346 bytes
                    h = LeCroy_Scope_Header(hdr_bytes)

                    if print_info:
                        print("dt =", h.dt)
                        print("t0 =", h.t0)
                        print("vertical_gain =", h.vertical_gain)
                        print("timebase =", h.timebase)
                        print("Input =", h.vertical_coupling)
                    
                    print(f"Header retrieved from Channel{channel}")
                    return h

            print("No channel with available header found.")
    except IOError:
        print(f"Cannot open file {file_path}.")
    except KeyError:
        print("LeCroy_Scope_Header info not available")
    return None

#======================================================================================
#======================================================================================

def read_channel_data(file_path, chnum, print_description=False): 
    """
    Reads the data of a specific channel from an HDF5 file.

    Parameters:
    file_path (str): The path to the HDF5 file.
    chnum (int): The channel number.
    print_description (bool, optional): Whether to print the channel description. Defaults to False.

    Returns:
    numpy.ndarray: The channel data.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            channel_path = f"Acquisition/LeCroy_scope/Channel{chnum}"
            if channel_path not in f:
                raise KeyError(f"Channel {chnum} does not exist in file {file_path}.")
            h5obj = f[channel_path]
            data = np.array(h5obj[()])
            channel_desc = channel_description(chnum, h5obj)
            if print_description:
                print(channel_desc)
    except (IOError, KeyError) as e:
        print(str(e))
        data = None
    return data

def read_position_data(file_path):
    """
    Reads the position data from an HDF5 file.

    Parameters:
    file_path (str): The path to the HDF5 file.

    Returns:
    tuple: A tuple containing the position array, xpos, ypos, and zpos.
    """
    with h5py.File(file_path, 'r') as f:
        pos_array = f["Control/Positions/positions_setup_array"][()]
        xpos = f["Control/Positions/positions_setup_array"].attrs['xpos']
        ypos = f["Control/Positions/positions_setup_array"].attrs['ypos']
        zpos = f["Control/Positions/positions_setup_array"].attrs['zpos']

    return pos_array, xpos, ypos, zpos

def generate_time_array(h, data):
    """
    Generates a time array based on the header information and data length.

    Parameters:
    h (LeCroy_Scope_Header): The header object.
    data (numpy.ndarray): The data array.

    Returns:
    numpy.ndarray: The time array.
    """
    dt = h.dt
    t0 = h.t0
    time_array = np.arrange(t0, t0 + len(data)*dt, dt)
