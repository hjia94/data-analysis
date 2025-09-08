# -*- coding: utf-8 -*-
# created by Jia Han, Sep-16-2019
# Read Lecroy scope data files in .trc (binary) or .txt (ascii)

'''
Note: when saving on scope, need to choose binary with word format

read_trc_data(file path) => signal, time array
	read binary file with open()
	first 11 bytes includes data_size information
	header information is encoded in the next 346 bytes which is decoded by LeCroy_Scope_Header.py
	data bytes are decoded using struct.unpack(). Each data point uses 2 bytes. Real voltage value is converted using header info
---------------------------------------------------------------------------
Note: when saving on scope, need to choose ascii and ',' as deliminator

read_txt_data(file path) => signal, time array
	use open() to look at first 5 rows and print out when and which scope data was taken from
	use numpy.loadtxt() to read the data and time array
---------------------------------------------------------------------------
compare_trigger_times(file path1, file path2) => True/False
	compare the trigger time information in the header of two .trc files
	return True if they are the same, False if not
--------------------------------------------------------------------------
read_hdf5_scope_tarr(f, scope_name) => time array
	Read the time array for a given scope group from an open HDF5 file.
--------------------------------------------------------------------------
read_hdf5_scope_data(f, scope_name, channel_name, shot_number) => voltage data array
	Read and convert raw scope channel data for a given shot from an open HDF5 file.
--------------------------------------------------------------------------
read_hdf5_all_scopes_channels(f, shot_number, include_tarr=True) => dict of all scope data
	Read all channel data for all scope groups for a given shot from an open HDF5 file
--------------------------------------------------------------------------

Feb.2024 update:
1. Variable Naming: Used snake_case for variable names, following PEP8 conventions.
2. Directly converted the unpacked data to a NumPy array and performed data manipulation in NumPy for efficiency.
3. Ensured consistent use of string formatting.

Sep.2025 update:
Added functions to read scope data from HDF5 files written by LAPD_DAQ, including reading time arrays and channel data for specific shots.
'''

import numpy as np
import struct
from datetime import datetime
import h5py

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

#======================================================================================

def read_hdf5_scope_tarr(f, scope_name):
	"""
	Read the time array for a given scope group from an open HDF5 file.

	Parameters
	----------
	f : h5py.File
		Open HDF5 file object (not a filename)
	scope_name : str
		Name of the scope group (e.g., 'bdotscope', 'xrayscope')

	Returns
	-------
	np.ndarray
		Time array for the specified scope group

	Raises
	------
	KeyError
		If the scope group or time array is not found
	"""
	if scope_name not in f:
		raise KeyError(f"Scope group '{scope_name}' not found in HDF5 file")
	scope_group = f[scope_name]
	if 'time_array' not in scope_group:
		raise KeyError(f"Time array not found for scope '{scope_name}'")
	return scope_group['time_array'][:]

#======================================================================================

def read_hdf5_scope_data(f, scope_name, channel_name, shot_number):
	"""
	Read and convert raw scope channel data for a given shot from an open HDF5 file.

	Parameters
	----------
	f : h5py.File
		Open HDF5 file object (not a filename)
	scope_name : str
		Name of the scope group (e.g., 'bdotscope', 'xrayscope')
	channel_name : str
		Name of the channel (e.g., 'C1', 'C2')
	shot_number : int
		Shot number to read (e.g., 1)

	Returns
	-------
	np.ndarray
		Calibrated voltage data for the specified channel

	Raises
	------
	KeyError
		If the group or dataset is missing
	ValueError
		If the shot is marked as skipped or header cannot be decoded
	"""
	
	# Fast local lookups
	try:
		scope_group = f[scope_name]
		shot_group = scope_group[f'shot_{shot_number}']
	except KeyError as e:
		raise KeyError(f"Missing group: {e}")

	attrs = shot_group.attrs
	if attrs.get('skipped', False):
		raise ValueError(f"Shot {shot_number} was skipped. Reason: {attrs.get('skip_reason', 'Unknown reason')}")

	data_key = f'{channel_name}_data'
	header_key = f'{channel_name}_header'
	try:
		raw_data = shot_group[data_key][:]
		header_bytes = shot_group[header_key][()]
	except KeyError as e:
		raise KeyError(f"Missing dataset: {e}")

	header = decode_header_info(header_bytes)
	if header is None:
		raise ValueError(f"Could not decode header for {scope_name}/shot_{shot_number}/{channel_name}")

	# Vectorized conversion
	gain = header.hdr.vertical_gain
	offset = header.hdr.vertical_offset
	voltage_data = raw_data.astype(np.float64) * gain - offset
	return voltage_data

#======================================================================================

def read_hdf5_all_scopes_channels(f, shot_number, include_tarr=True):
	"""
	Read all channel data for all scope groups for a given shot from an open HDF5 file.

	Parameters
	----------
	f : h5py.File
		Open HDF5 file object (not a filename)
	shot_number : int
		Shot number to load (e.g., 1 => group 'shot_1')
	include_tarr : bool, optional
		If True, include the scope time array in the result under 'time_array'.
		If False, the 'time_array' value will be None. Default True.

	Returns
	-------
	dict
		Nested dictionary of the form:
		{
		  scope_name: {
			'time_array': np.ndarray | None,
			'channels': {
			   channel_name: np.ndarray  # voltage data
			}
		  },
		  ...
		}
	"""
	result = {}

	skip_groups = {'Configuration', 'Control'}
	for scope_name, scope_group in f.items():
		if scope_name in skip_groups:
			continue
		shot_group_name = f'shot_{shot_number}'
		if shot_group_name not in scope_group:
			print(f"Scope '{scope_name}' is not recorded for shot '{shot_number}'")
			continue
		shot_group = scope_group[shot_group_name]
		attrs = shot_group.attrs
		if attrs.get('skipped', False):
			print(f"Shot {shot_number} for scope '{scope_name}' was skipped: {attrs.get('skip_reason', 'Unknown reason')}")
			continue
		if include_tarr:
			try:
				tarr = read_hdf5_scope_tarr(f, scope_name)
				result[scope_name] = {'time_array': tarr}
			except Exception as e:
				print(f"Could not read time array for scope '{scope_name}': {e}")
				tarr = None
		channels = {}
		for key, ds in shot_group.items():
			if not (isinstance(ds, h5py.Dataset) and key.endswith('_data')):
				continue
			channel_name = key[:-5]
			channels[channel_name] = read_hdf5_scope_data(f, scope_name, channel_name, shot_number)
		result[scope_name]['channels'] = channels

	return result
