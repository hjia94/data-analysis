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

import os
import sys
import functools
import numpy as np
import h5py

from .LeCroy_Scope_Header import LeCroy_Scope_Header

#======================================================================================
# scope_io locator
#======================================================================================
# Manual override for where the sibling LAPD_DAQ repo lives, used only if a plain
# `import scope_io` fails (i.e. LAPD_DAQ is not pip-installed).  Leave as None to
# rely on installed `scope_io` / the sibling-clone fallback; set it to the
# LAPD_DAQ repo root (e.g. r"C:\Users\hjia9\Documents\GitHub\LAPD_DAQ") to point
# at a clone elsewhere.
LAPD_DAQ_PATH = None

# The HDF5 scope readers below are thin wrappers over the `scope_io` package that
# lives in the sibling LAPD_DAQ repo (single source of truth for LAPD_DAQ HDF5
# decoding). `scope_io` ships with the `lapd-daq` package, so the normal path is a
# plain `import scope_io` once `pip install -e ../LAPD_DAQ` (the `scope` extra) is
# in place. Sibling-clone path discovery is the fallback for checkouts run
# without a `pip install` (common in this repo), not a temporary measure.
# The import is done lazily, inside the HDF5 readers, so the legacy .trc/.txt
# readers in this module keep working even if LAPD_DAQ is not present at all.

@functools.lru_cache(maxsize=None)
def _import_scope_io():
	"""Import LAPD_DAQ's `scope_io` package (single source of truth for LAPD_DAQ
	HDF5 scope decoding).

	Resolution order:
	  1. installed `scope_io` (from `pip install -e ../LAPD_DAQ`) -- the normal path
	  2. fallback: sibling-clone discovery via the module-level LAPD_DAQ_PATH
	     constant (edit it at the top of this file), else `../../LAPD_DAQ` relative
	     to this file (data-analysis and LAPD_DAQ as sibling clones), put on
	     sys.path -- for checkouts run without an install.

	Raises a clear ImportError if neither resolves. Cached so the lookup runs at
	most once per process.
	"""
	try:
		import scope_io
		return scope_io
	except ImportError:
		pass

	# Fallback: discover a sibling LAPD_DAQ clone and put it on sys.path.
	candidates = []
	if LAPD_DAQ_PATH:
		candidates.append(LAPD_DAQ_PATH)
	# src/data_analysis/io/scope_reader.py -> data_analysis -> src -> data-analysis -> GitHub -> LAPD_DAQ
	candidates.append(os.path.abspath(os.path.join(
		os.path.dirname(__file__), '..', '..', '..', '..', 'LAPD_DAQ')))

	for path in candidates:
		if os.path.isdir(os.path.join(path, 'scope_io')) and path not in sys.path:
			sys.path.insert(0, path)

	try:
		import scope_io
		return scope_io
	except ImportError:
		raise ImportError(
			"Could not import 'scope_io': install LAPD_DAQ (`pip install -e "
			"../LAPD_DAQ`, the `scope` extra), or clone LAPD_DAQ beside "
			"data-analysis (as a sibling folder), or set the LAPD_DAQ_PATH "
			"constant at the top of scope_reader.py to the LAPD_DAQ repo root.")

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

	Delegates to ``scope_io.read_hdf5_scope_tarr`` (LAPD_DAQ); see :func:`_import_scope_io`.
	Returns the scope group's time array. Raises KeyError if the scope group or
	its time array is missing.
	"""
	scope_io = _import_scope_io()
	return scope_io.read_hdf5_scope_tarr(f, scope_name)

#======================================================================================

def read_hdf5_scope_data(f, scope_name, channel_name, shot_number):
	"""
	Read and convert raw scope channel data for a given shot from an open HDF5 file.

	Delegates to ``scope_io.read_hdf5_scope_data`` (LAPD_DAQ); see
	:func:`_import_scope_io`. Returns ``(voltage_data, dt, t0)`` with the voltage
	scaled to volts. Raises KeyError if the group/dataset is missing and ValueError
	if the shot is skipped or the header cannot be decoded.
	"""
	scope_io = _import_scope_io()
	return scope_io.read_hdf5_scope_data(f, scope_name, channel_name, shot_number)

#======================================================================================

def scope_shot_numbers(scope_group):
	"""
	Return the available shot numbers in a scope group (an open HDF5 group).

	Delegates to ``scope_io.scope_shot_numbers`` (LAPD_DAQ); see :func:`_import_scope_io`.
	"""
	scope_io = _import_scope_io()
	return scope_io.scope_shot_numbers(scope_group)

#======================================================================================

def read_hdf5_scope_channel_shots(f, scope_name, channel_name, shot_numbers, expected_len=None):
	"""
	Read many shots of one channel into a ``(nshot, nsamples)`` float64 stack.

	Delegates to ``scope_io.read_hdf5_scope_channel_shots`` (LAPD_DAQ); see
	:func:`_import_scope_io`. Returns ``(stack, dt, t0)`` (NaN rows for unreadable
	shots; ``None`` stack if no shot could be read).
	"""
	scope_io = _import_scope_io()
	return scope_io.read_hdf5_scope_channel_shots(f, scope_name, channel_name, shot_numbers, expected_len=expected_len)

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
		result[scope_name] = {}
		shot_group_name = f'shot_{shot_number}'
		if shot_group_name not in scope_group:
			print(f"Scope '{scope_name}' is not recorded for shot '{shot_number}'")
			continue
		shot_group = scope_group[shot_group_name]
		attrs = shot_group.attrs
		if attrs.get('skipped', False):
			print(f"Shot {shot_number} for scope '{scope_name}' was skipped: {attrs.get('skip_reason', 'Unknown reason')}")
			continue

		channels = {}
		for key, ds in shot_group.items():
			if not (isinstance(ds, h5py.Dataset) and key.endswith('_data')):
				continue
			channel_name = key[:-5]
			data, dt, t0 = read_hdf5_scope_data(f, scope_name, channel_name, shot_number)
			channels[channel_name] = data
		result[scope_name]['channels'] = channels

		if include_tarr:
			try:
				tarr = read_hdf5_scope_tarr(f, scope_name)
				if len(tarr) != len(data):
					tarr = np.arange(len(data)) * dt + t0
				result[scope_name]['time_array'] = tarr
			except Exception as e:
				print(f"Could not read time array for scope '{scope_name}': {e}")
				result[scope_name]['time_array'] = None

	return result

#======================================================================================

def read_scope_channel_descriptions(f, scope_name):
	"""
	Return a dictionary of channel descriptions for a given scope group from an open HDF5 file.

	Delegates to ``scope_io.read_hdf5_scope_channel_descriptions`` (LAPD_DAQ),
	which handles both the new (per-channel ``<CH>_description`` scope-group
	attrs) and old (per-shot ``<CH>_data`` description attr) layouts, and scans
	for the first populated shot rather than assuming ``shot_1`` exists. Returns
	``{}`` if the scope group is absent. See :func:`_import_scope_io`.
	"""
	scope_io = _import_scope_io()
	return scope_io.read_hdf5_scope_channel_descriptions(f, scope_name)
