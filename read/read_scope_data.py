# -*- coding: utf-8 -*-
# created by Jia Han, Sep-16-2019
# Read Lecroy scope data files in .trc (binary) or .txt (ascii)

'''
read_trc_data(file path) => signal, time array
	read binary file with open()
	first 11 bytes includes data_size information
	header information is encoded in the next 346 bytes which is decoded by LeCroy_Scope_Header.py
	data bytes are decoded using struct.unpack(). Each data point uses 2 bytes. Real voltage value is converted using header info

Note: when saving on scope, need to choose binary with word format

--------------------------------------------------------------------

read_txt_data(file path) => signal, time array
	use open() to look at first 5 rows and print out when and which scope data was taken from
	use numpy.loadtxt() to read the data and time array

Note: when saving on scope, need to choose ascii and ',' as deliminator

--------------------------------------------------------------------------
--------------------------------------------------------------------------
Feb.2024 update:
1. Variable Naming: Used snake_case for variable names, following PEP8 conventions.
2. Directly converted the unpacked data to a NumPy array and performed data manipulation in NumPy for efficiency.
3. Ensured consistent use of string formatting.

'''

import numpy as np
import struct
from datetime import datetime

from LeCroy_Scope_Header import LeCroy_Scope_Header

#======================================================================================

def decode_header_info(hdr_bytes):
    try:
        header = LeCroy_Scope_Header(hdr_bytes)
    except Exception as e:
        print("Error decoding LeCroy_Scope_Header info:", e)
        header = None
    return header

#======================================================================================

def get_trigger_time(file_path):
    """
    Extract trigger timing information from LeCroy scope file header.
    
    Parameters:
    -----------
    file_path : str
        Path to the .trc file
        
    Returns:
    --------
    datetime
        datetime object of when the trace was triggered
    """
    
    with open(file_path, mode='rb') as file:
        file_content = file.read()
    
    first_11 = file_content[:11].decode()
    if not first_11.startswith('#9'):
        raise SyntaxError('First two bytes are not #9')
    
    hdr_bytes = file_content[11:11+346]
    header = decode_header_info(hdr_bytes)
    
    if header is None:
        raise ValueError("Could not decode header information")
    
    # Extract trigger time components from header
    # Note: LeCroy stores year as offset from 1900
    year = header.hdr.tt_year
    month = header.hdr.tt_months
    day = header.hdr.tt_days
    hour = header.hdr.tt_hours
    minute = header.hdr.tt_minute
    second = header.hdr.tt_second
    
    
    return {'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second}

#======================================================================================

def compare_trigger_times(file_path1, file_path2, debug=False):
    """
    Compare trigger times of two LeCroy scope files.
    
    Parameters:
    -----------
    file_path1 : str
        Path to the first .trc file
    file_path2 : str
        Path to the second .trc file
    tolerance_seconds : float, optional
        Maximum difference in seconds to consider times as "same" (default: 1 second)
        
    Returns:
    --------
    bool
        True if trigger times are the same (within tolerance), False otherwise
    """
    
    try:
        time1 = get_trigger_time(file_path1)
        time2 = get_trigger_time(file_path2)
        
        if time1['year'] != time2['year']: 
            return False
        elif time1['month'] != time2['month']: 
            return False
        elif time1['day'] != time2['day']: 
            return False
        elif time1['hour'] != time2['hour']: 
            return False
        elif time1['minute'] != time2['minute'] or time1['second'] != time2['second']:
            if debug:
                # Calculate total seconds for both times (minute*60 + second)
                total_seconds1 = time1['minute'] * 60 + time1['second']
                total_seconds2 = time2['minute'] * 60 + time2['second']
                diff_seconds = total_seconds2 - total_seconds1
                print(f'Time difference: {diff_seconds} seconds')
            return False
        else: 
            return True
    except Exception as e:
        print(f"Error comparing trigger times: {e}")
        return False

#======================================================================================

def read_trc_data(file_path, list_some_header_info=False):

	with open(file_path, mode='rb') as file: # rb -> read binary
		file_content = file.read()
	
	first_11 = file_content[:11].decode()

	if not first_11.startswith('#9'):
		raise SyntaxError('First two bytes are not #9')

	hdr_bytes = file_content[11:11+346]
	header = decode_header_info(hdr_bytes)

	data_size = int( (int(first_11[2:]) - 346) / 2)
	if data_size != len(header.time_array):
		print('Time array length from header %i does not equal %i from first 11 bytes' %(len(header.time_array), data_size))
	data_size = len(header.time_array)

	if list_some_header_info:
		print("dt =", header.dt)
		print("t0 =", header.t0)
		print("vertical_gain =", header.vertical_gain)
		print("timebase =", header.timebase)
		print("Input = ", header.vertical_coupling)

	data_bytes = file_content[11+346:]

	print('Reading data...')
	fmt = f"={data_size}h"
	data = np.frombuffer(data_bytes, dtype=fmt)
	data = data[0,:] * header.hdr.vertical_gain - header.hdr.vertical_offset

	print('Done')

	return data, header.time_array # signal, time array

#======================================================================================
def read_trc_data_simplified(file_path):

	with open(file_path, mode='rb') as file: # rb -> read binary
		file_content = file.read()

	hdr_bytes = file_content[11:11+346]
	header = decode_header_info(hdr_bytes)
	data_size = len(header.time_array)
	
	data_bytes = file_content[11+346:]
	fmt = f"={data_size}h"
	data = np.frombuffer(data_bytes, dtype=fmt)
	data = data[0,:] * header.hdr.vertical_gain - header.hdr.vertical_offset

	return data, header.time_array # signal, time array

#======================================================================================

def read_txt_data(ifn):

	with open(ifn, "r") as file:
		file_content = file.read()
		
		if 'Segment' not in file_content[:50]:
			print('First 5 rows might include data. Check on text reader before using this function to read.')

		print(file_content[:15], ' trace saved on', file_content[100:119])


	data = np.loadtxt(ifn,dtype=float, delimiter=',', skiprows=5)

	print('Done')

	return data[:,1], data[:,0] - data[0,0] # signal, time array
