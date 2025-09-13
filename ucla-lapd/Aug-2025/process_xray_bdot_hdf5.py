#!/usr/bin/env python3
"""
Script to process X-ray and Bdot data, creating combined plots for each shot.
Each figure will contain 4 subplots:
1. Raw X-ray signal with detected pulses
2. Photon counts histogram
3. Bdot STFT spectrogram
4. Combined STFT and counts plot
"""

import os
import sys
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import colors
import matplotlib.colors as colors  # Ensure we have colors for LogNorm
from scipy import ndimage
import tkinter as tk
import cv2
import re
import h5py
import glob

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__ if '__file__' in globals() else os.getcwd()), '../..'))
sys.path = [repo_root, f"{repo_root}/read", f"{repo_root}/object_tracking"] + sys.path

from read.read_scope_data import read_trc_data, read_hdf5_all_scopes_channels, read_scope_channel_descriptions
from data_analysis_utils import Photons, calculate_stft, counts_per_bin
from object_tracking.read_cine import read_cine, convert_cine_to_avi
from object_tracking.track_object import track_object, detect_chamber, get_vel_freefall, get_pos_freefall

#===========================================================================================================
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

# Simple prefixed logger for clearer terminal output
def _log(tag, msg):
	print(f"[{tag}] {msg}")

#===========================================================================================================

def get_magnetron_power_data(f, result, scope_name='magscope'):
	"""
	Calculate magnetron power from HDF5 file data.
	"""
	if scope_name not in result:
		_log('POWER', f"Scope '{scope_name}' not found.")
		return None, None, None, None, None

	tarr = result[scope_name].get('time_array')
	chan_data = result[scope_name].get('channels', {})
	descriptions = read_scope_channel_descriptions(f,'magscope')

	I_data = None
	V_data = None
	Pref_data = None

	for ch, desc in descriptions.items():
		if 'current' in desc:
			I_data = chan_data[ch]
			if isinstance(desc, str):
				m = re.search(r'(\d+\.?\d*)\s*a/v', desc.lower())
				if m:
					scale_factor = float(m.group(1))
					I_data = I_data * scale_factor

		if 'voltage' in desc:
			V_data = chan_data[ch]
		if 'pref' in desc:
			Pref_data = chan_data[ch]

	P_data = None
	if I_data is not None and V_data is not None:
		P_data = ndimage.gaussian_filter1d(I_data * (-V_data) * 0.6, sigma=100)
		_log('POWER', "Magnetron power calculated")
	else:
		_log('POWER', "Cannot calculate power: missing current or voltage data")

	return tarr, P_data

def get_xray_data(result, scope_name = 'xrayscope'):
	tarr_x = result[scope_name].get('time_array')
	xray_data = result[scope_name]['channels']['C2']
	return tarr_x, xray_data

def get_bdot_data(f, result, scope_name='bdotscope'):

	tarr = result[scope_name].get('time_array')
	chan_data = result[scope_name].get('channels', {})
	descriptions = read_scope_channel_descriptions(f, scope_name)

	# for ch, desc in descriptions.items():

	#     m = re.search(r'P(\d+)', desc)
	#     if m:
	#         p_number = int(m.group(1))

	return tarr, chan_data, descriptions


def calculate_bdot_stft(tarr, bdot_data, freq_bins=1000, overlap_fraction=0.05, freq_min=200e6, freq_max=2000e6):
	'''
	Calculates STFT for each Bdot signal in bdot_data.
	
	Parameters:
	- tarr: Time array for Bdot signals from get_bdot_data
	- bdot_data: Dictionary of Bdot data channels from get_bdot_data
	- freq_bins: Number of frequency bins for STFT
	- overlap_fraction: Fraction of overlap for STFT windows
	- freq_min: Minimum frequency to include in STFT
	- freq_max: Maximum frequency to include in STFT
	
	Returns: 
	- stft_time: Time array for STFT
	- freq: Frequency array for STFT
	- stft_matrices: Dictionary of all STFT matrices by channel name
	'''
	stft_matrices = {}
	stft_time = None
	freq = None
	
	# Process each channel in bdot_data
	for channel, data in bdot_data.items():
		if data is not None:
			freq, stft_matrix, stft_time = calculate_stft(tarr, data, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
			stft_matrices[channel] = stft_matrix
			_log('STFT', f"Calculated STFT for channel {channel}")
		else:
			stft_matrices[channel] = None
			_log('STFT', f"Skipped STFT for channel {channel} (no data)")
	

	return stft_time, freq,  stft_matrices

def process_bdot(ifn, freq_bins=1000, overlap_fraction=0.05, freq_min=50e6, freq_max=1000e6, plot=True):
	"""
	Process Bdot data from an HDF5 file by averaging STFT across all shots for each channel.
	
	Parameters:
	- ifn: Path to the HDF5 file
	- freq_bins: Number of frequency bins for STFT
	- overlap_fraction: Fraction of overlap for STFT windows
	- freq_min: Minimum frequency to include in STFT (Hz)
	- freq_max: Maximum frequency to include in STFT (Hz)
	- plot: Whether to create and display plots
	
	Returns:
	- Dictionary of averaged STFT matrices by channel
	- Dictionary of channel descriptions
	- Time array for STFT
	- Frequency array for STFT
	"""
	_log('BDOT', f"Processing Bdot data from {os.path.basename(ifn)}")
	
	# Open the HDF5 file
	with h5py.File(ifn, 'r') as f:
		# Get the shot numbers
		shot_numbers = f['Control/FastCam']['shot number'][()]
		_log('BDOT', f"Found {len(shot_numbers)} shots")
		
		# Initialize dictionaries to store STFT matrices and descriptions
		all_stft_matrices = {}  # Format: {channel: [stft_matrix1, stft_matrix2, ...]}
		stft_tarr_final = None
		freq_arr_final = None
		
		# Process each shot
		for shot_num in shot_numbers:
			_log('BDOT', f"Processing shot {shot_num}")
			
			# Get the Bdot data for this shot
			result = read_hdf5_all_scopes_channels(f, shot_num)
			tarr_B, bdot_data, descriptions = get_bdot_data(f, result)
			
			if tarr_B is None or len(bdot_data) == 0:
				_log('BDOT', f"No Bdot data for shot {shot_num}")
				continue
			
			# Calculate STFT for each channel
			stft_tarr, freq_arr, stft_matrices = calculate_bdot_stft(tarr_B, bdot_data, freq_bins, overlap_fraction, freq_min, freq_max)
			
			# Store the results
			for channel, matrix in stft_matrices.items():
				if channel not in all_stft_matrices:
					all_stft_matrices[channel] = []
				if matrix is not None:
					all_stft_matrices[channel].append(matrix)
			
			# Store the time and frequency arrays (assumed to be the same for all shots)
			if stft_tarr is not None and freq_arr is not None:
				stft_tarr_final = stft_tarr
				freq_arr_final = freq_arr
	
	# Calculate average STFT for each channel
	avg_stft_matrices = {}
	for channel, matrices in all_stft_matrices.items():
		if matrices:  # If there are any valid matrices for this channel
			avg_stft_matrices[channel] = np.mean(np.array(matrices), axis=0)
			_log('BDOT', f"Averaged {len(matrices)} STFT matrices for channel {channel}")
	

	if plot and avg_stft_matrices:
		plot_averaged_bdot_stft(avg_stft_matrices, descriptions, stft_tarr_final, freq_arr_final)


def process_shot_xray(tarr_x, xray_data, min_ts, d, threshold, debug=False):
	"""Process a single shot of xray data and return data for averaging.
	"""

	detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)
	detector.reduce_pulses()

	return detector.pulse_times, detector.pulse_amplitudes

def process_video(base_dir, cam_file, tracking_dict={}, tracking_file=None, plotting=False):
	'''
	Process the video file for a given shot number and return the time at which the ball reaches the chamber center
	'''
	
	filepath = os.path.join(base_dir, cam_file)
	if not os.path.exists(filepath):
		raise FileNotFoundError("Video file does not exist.")
	avi_path = os.path.join(base_dir, f"{os.path.splitext(cam_file)[0]}.avi")

	# Check if we already have results for this file
	if filepath in tracking_dict:
		_log('VIDEO', f"Loading existing tracking results for {filepath}")
		cf, ct = tracking_dict[filepath]
		if cf is None or ct is None:
			_log('VIDEO', "No object was tracked in previous analysis.")
			return None
	else:
		_log('VIDEO', "No existing tracking results found. Processing video...")
		tarr, frarr, dt = read_cine(filepath)

		# convert cine to avi if not already done
		if not os.path.exists(avi_path):
			_log('VIDEO', f"Converting {filepath} to {avi_path}")
			convert_cine_to_avi(frarr, avi_path)

		_log('VIDEO', f"Tracking object in {avi_path}")
		parr, frarr, cf = track_object(avi_path)

		if cf is None:
			_log('VIDEO', "No object tracked in video.")
			ct = None
			# Save result with None frame
			try:
				tracking_dict[filepath] = (cf, ct)
				if tracking_file is not None:
					np.save(tracking_file, tracking_dict)
			except Exception as e:
				_log('VIDEO', f"Error saving tracking results: {e}")
			return ct
		else:
			ct = tarr[cf]
			try:
				tracking_dict[filepath] = (cf, ct)         # Add new result to the dictionary and save
				np.save(tracking_file, tracking_dict)
			except Exception as e:
				_log('VIDEO', f"Error saving tracking results: {e}")

	if plotting and cf is not None:
		cap = cv2.VideoCapture(avi_path)
		cap.set(cv2.CAP_PROP_POS_FRAMES, cf)
		ret, frame = cap.read()
		if not ret:
			raise ValueError(f"Could not read frame")
		cap.release()

		fig, ax = plt.subplots(figsize=(8, 8))
		ax.set_title(cam_file)

		ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		(cx, cy), chamber_radius = detect_chamber(frame)
		chamber_circle = plt.Circle((cx, cy), chamber_radius, fill=False, color='green', linewidth=2)
		ax.add_patch(chamber_circle)
		_log('VIDEO', f"ball reaches chamber center at t={ct * 1e3:.3f}ms from plasma trigger")
		ax.axis('off')
		plt.draw()
		plt.pause(0.1)

	return ct

def plot_averaged_bdot_stft(stft_matrices, description, stft_tarr, freq_arr):
	"""
	Plot averaged Bdot STFT data for each channel.
	
	Parameters:
	- stft_matrices: Dictionary of averaged STFT matrices by channel
	- channel_description: String description for the channel
	- stft_tarr: Time array for STFT (seconds)
	- freq_arr: Frequency array for STFT (Hz)
	"""
	# Get the number of channels
	num_channels = len(stft_matrices)
	if num_channels == 0:
		_log('PLOT', "No STFT matrices to plot")
		return None
	
	# Create figure with subplots
	fig, axes = plt.subplots(num_channels, 1, figsize=(8,8), 
							num="Averaged_Bdot_STFT", sharex=True)
	
	# Handle the case of a single channel (axes is not an array)
	if num_channels == 1:
		axes = [axes]
	
	# Sort the channels for consistent display
	channels = sorted(stft_matrices.keys())
	
	# Plot each STFT matrix
	for i, channel in enumerate(channels):
		matrix = stft_matrices[channel]
		
		# Create the plot - ensure matrix has positive values for LogNorm
		positive_matrix = matrix.copy()
		min_positive = positive_matrix[positive_matrix > 0].min() if np.any(positive_matrix > 0) else 1e-10
		positive_matrix[positive_matrix <= 0] = min_positive
		
		im = axes[i].imshow(positive_matrix.T,
						  aspect='auto',
						  origin='lower',
						  extent=[stft_tarr[0]*1e3, stft_tarr[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
						  interpolation='None',
						  cmap='jet',
						  norm=colors.LogNorm(vmin=min_positive, 
										  vmax=positive_matrix.max()))
		
		axes[i].set_ylabel('Frequency (MHz)')
		axes[i].set_title(description[channel])
		fig.colorbar(im, ax=axes[i], label='Magnitude')
	
	# Set common x-axis label
	axes[-1].set_xlabel('Time (ms)')
	plt.show(block=True)

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

def xray_wt_cam(base_dir, fn):
	# Define path
	ifn = os.path.join(base_dir, fn)
	
	# Extract two-digit prefix from filename (e.g., "02" from "02_He1kG430G_5800A_K-25_2025-08-12.hdf5")
	file_prefix = fn[:2]

	# Initialize or load the analysis dictionary
	analysis_file = os.path.join(base_dir, 'analysis_results.npy')
	if os.path.exists(analysis_file):
		analysis_dict = np.load(analysis_file, allow_pickle=True).item()
	else:
		analysis_dict = {}
		
	# Initialize or load the tracking result dictionary
	tracking_file = os.path.join(base_dir, f"tracking_result.npy")
	if os.path.exists(tracking_file):
		tracking_dict = np.load(tracking_file, allow_pickle=True).item()
	else:
		tracking_dict = {}

	with h5py.File(ifn, 'r') as f:
		_log('FILE', f"Opened file {ifn} for processing.")
		shot_numbers = f['Control/FastCam']['shot number'][()]

	for shot_num in shot_numbers:
		_log('SHOT', f"Processing shot {shot_num}")
		analysis_key = f"{file_prefix}_{shot_num:03d}"

		with h5py.File(ifn, 'r') as f:
			cine_narr = f['Control/FastCam/cine file name'][()]
			cam_file = cine_narr[shot_num-1].decode('utf-8')
			if shot_num != int(cam_file[-8:-5]):
				_log('FILE', f"Warning: shot number {shot_num} does not match {cam_file}")
			mfn = os.path.join(base_dir, cam_file)

		if mfn not in tracking_dict.keys():
			_log('TRACKING', f"Analyzing video {cam_file} for tracking...")
			try:
				t0 = process_video(base_dir, cam_file, tracking_dict=tracking_dict, tracking_file=tracking_file)
			except ValueError as e:
				_log('VIDEO', f"Error processing video for shot {shot_num}: {e}")
				continue
			except FileNotFoundError as e:
				_log('VIDEO', f"No video file found for shot {shot_num}; skipping...")
				continue

		if analysis_key in analysis_dict:
			_log('ANALYSIS', f"Analysis result exists for {analysis_key}")
		else:
			with h5py.File(ifn, 'r') as f:
				result = read_hdf5_all_scopes_channels(f, shot_num, include_tarr=True)
				# tarr_P, P_data = get_magnetron_power_data(f, result)
				tarr_x, xray_data = get_xray_data(result)
				threshold = [5, 70]
				min_ts = 0.8e-6
				d = 0.1
				pulse_tarr, pulse_amp = process_shot_xray(tarr_x, xray_data, min_ts, d, threshold)

				analysis_dict[analysis_key] = (pulse_tarr, pulse_amp)
				try:
					np.save(analysis_file, analysis_dict)
					_log('ANALYSIS', f"Added new analysis result for {analysis_key}")
				except Exception as e:
					_log('ANALYSIS', f"Error saving analysis results: {e}")


def batch_process_xray(base_dir):
	"""
	Batch process all HDF5 files in base_dir using xray_wt_cam,
	skipping files whose prefix already exists in analysis_results.npy.
	"""

	# Find all .hdf5 files in base_dir
	hdf5_files = [f for f in os.listdir(base_dir) if f.endswith('.hdf5')]

	for fn in hdf5_files:
		print(f"Processing {fn} ...")
		xray_wt_cam(base_dir, fn)


#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

	base_dir = r"F:\AUG2025\P23"
	batch_process_xray(base_dir)

	# ifn = r"F:\AUG2025\P24\27_He1kG430G_5800A_K-5_2025-08-12.hdf5"
	# process_bdot(ifn)