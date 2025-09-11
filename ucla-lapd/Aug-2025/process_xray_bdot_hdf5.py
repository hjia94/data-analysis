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

	return tarr, chan_data


def read_bdot_signals_from_hdf5(hdf5_filename, debug=False):
	"""Minimal reader for Bdot signals to support processing.
	Returns (tarr_B, bdot_data, channel_info) or (None, {}, {}) on failure.
	"""
	try:
		with h5py.File(hdf5_filename, 'r') as f:
			if 'bdotscope' not in f:
				return None, {}, {}
			scope = f['bdotscope']
			tarr_B = scope.get('time_array')[()]
			bdot_data = {}
			if 'channels' in scope:
				for ch in scope['channels']:
					bdot_data[ch] = scope['channels'][ch][()]
			try:
				channel_info = read_scope_channel_descriptions(f, 'bdotscope')
			except Exception:
				channel_info = {}
			return tarr_B, bdot_data, channel_info
	except Exception as e:
		if debug:
			_log('BDOT', f"Error reading {hdf5_filename}: {e}")
		return None, {}, {}

def calculate_bdot_stft(tarr_B, bdot_data, channel_info, freq_bins=1000, overlap_fraction=0.05, freq_min=200e6, freq_max=2000e6):
	'''
	Calculates STFT for each Bdot signal in bdot_data.
	Returns: stft_time, freq, stft_matrix1, stft_matrix2, stft_matrix3, stft_matrices
	'''
	stft_matrices = {}
	stft_time = None
	freq = None
	for channel, data in bdot_data.items():
		if data is not None:
			freq, stft_matrix, stft_time = calculate_stft(tarr_B, data, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
			stft_matrices[channel] = stft_matrix
		else:
			stft_matrices[channel] = None
	channels = sorted([ch for ch in channel_info.keys() if ch.startswith('C')])
	stft_matrix1 = stft_matrices.get(channels[0]) if len(channels) > 0 else None
	stft_matrix2 = stft_matrices.get(channels[1]) if len(channels) > 1 else None
	stft_matrix3 = stft_matrices.get(channels[2]) if len(channels) > 2 else None
	return stft_time, freq, stft_matrix1, stft_matrix2, stft_matrix3, stft_matrices

def process_shot_bdot(file_number, base_dir, debug=False):
	"""
	Process Bdot signals for a specific shot number.
	Returns the STFT time array, frequency array, and STFT matrices for three channels.
	
	Args:
		file_number: The shot number to process
		base_dir: Base directory containing the HDF5 files
		debug: Whether to print debug information
		
	Returns:
		stft_tarr: Time array for the STFT
		freq_arr: Frequency array for the STFT
		stft_matrix1, stft_matrix2, stft_matrix3: STFT matrices for three channels
	"""
	# Construct the HDF5 filename
	hdf5_filename = os.path.join(base_dir, f'{file_number}.hdf5')
	
	if not os.path.exists(hdf5_filename):
		print(f"Error: HDF5 file not found: {hdf5_filename}")
		return None, None, None, None, None
	
	# Read the Bdot signals
	tarr_B, bdot_data, channel_info = read_bdot_signals_from_hdf5(hdf5_filename, debug=debug)
	
	if tarr_B is None or len(bdot_data) == 0:
		print(f"Error: Failed to read Bdot signals from {hdf5_filename}")
		return None, None, None, None, None
	
	# Calculate the STFT for each channel
	stft_tarr, freq_arr, stft_matrices = calculate_bdot_stft(tarr_B, bdot_data, channel_info)
	
	# Extract the STFT matrices for the three channels (assuming they exist)
	channels = list(stft_matrices.keys())
	if len(channels) >= 3:
		stft_matrix1 = stft_matrices[channels[0]]
		stft_matrix2 = stft_matrices[channels[1]]
		stft_matrix3 = stft_matrices[channels[2]]
	else:
		# Handle the case where we don't have three channels
		stft_matrix1 = stft_matrices.get(channels[0], None) if channels else None
		stft_matrix2 = stft_matrices.get(channels[1], None) if len(channels) > 1 else None
		stft_matrix3 = stft_matrices.get(channels[2], None) if len(channels) > 2 else None
	
	return stft_tarr, freq_arr, stft_matrix1, stft_matrix2, stft_matrix3

def process_shot_xray(tarr_x, xray_data, min_ts, d, threshold, debug=False):
	"""Process a single shot of xray data and return data for averaging.
	"""

	detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)
	detector.reduce_pulses()

	return detector.pulse_times, detector.pulse_amplitudes

def process_video(base_dir, cam_file):
	'''
	Process the video file for a given shot number and return the time at which the ball reaches the chamber center
	'''
	
	filepath = os.path.join(base_dir, cam_file)
	if not os.path.exists(filepath):
		raise FileNotFoundError("Video file does not exist.")
	avi_path = os.path.join(base_dir, f"{os.path.splitext(cam_file)[0]}.avi")

	# Initialize or load the tracking result dictionary
	tracking_file = os.path.join(base_dir, f"tracking_result.npy")
	if os.path.exists(tracking_file):
		tracking_dict = np.load(tracking_file, allow_pickle=True).item()
	else:
		tracking_dict = {}

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

	# For plotting
	if cf is not None:
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

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

def xray_wt_cam(base_dir, fn):
	# Define path
	ifn = os.path.join(base_dir, fn)
	analysis_file = os.path.join(base_dir, 'analysis_results.npy')
	
	# Extract two-digit prefix from filename (e.g., "02" from "02_He1kG430G_5800A_K-25_2025-08-12.hdf5")
	file_prefix = fn[:2]
	
	# Initialize or load the analysis dictionary
	analysis_dict = {}
	if os.path.exists(analysis_file):
		try:
			analysis_dict = np.load(analysis_file, allow_pickle=True).item()
		except Exception as e:
			_log('ANALYSIS', f"Error loading analysis results: {e}")
			analysis_dict = {}
	
	with h5py.File(ifn, 'r') as f:
		shot_numbers = f['Control/FastCam']['shot number'][()]

	for shot_num in shot_numbers:
		_log('SHOT', f"Processing shot {shot_num}")

		with h5py.File(ifn, 'r') as f:
			_log('FILE', f"Reading data from {ifn}")
			cine_narr = f['Control/FastCam/cine file name'][()]
			cam_file = cine_narr[shot_num-1]
			if shot_num != int(cam_file[-8:-5]):
				_log('FILE', f"Warning: shot number {shot_num} does not match {cam_file}")

			result = read_hdf5_all_scopes_channels(f, shot_num, include_tarr=True)
			tarr_P, P_data = get_magnetron_power_data(f, result)
			tarr_x, xray_data = get_xray_data(result)

		try:
			t0 = process_video(base_dir, cam_file.decode('utf-8'))
		except FileNotFoundError as e:
			_log('VIDEO', f"No video file found for shot {shot_num}; skipping...")
			continue

		# Plot power data if available
		if P_data is not None and tarr_P is not None:
			fig, ax = plt.subplots(figsize=(15, 5), num=f"shot_{shot_num}")
			ax.plot(tarr_P*1e3, P_data*1e-4, 'b-', linewidth=2)
			ax.set_xlabel('Time (ms)')
			ax.set_ylabel('Power (kW)')
			ax.grid(True)
			plt.tight_layout()
			plt.draw()
			plt.pause(0.1)
		

		# Create composite key: file_prefix + shot_number (e.g., "02_001", "02_002")
		analysis_key = f"{file_prefix}_{shot_num:03d}"
		if analysis_key in analysis_dict:
			pulse_tarr, pulse_amp = analysis_dict[analysis_key]
			_log('ANALYSIS', f"Using cached analysis for {analysis_key}")
		else:
			threshold = [5, 70]
			min_ts = 0.8e-6
			d = 0.1
			pulse_tarr, pulse_amp = process_shot_xray(tarr_x, xray_data, min_ts, d, threshold)

			# Save new results with composite key
			analysis_dict[analysis_key] = (pulse_tarr, pulse_amp)
			try:
				np.save(analysis_file, analysis_dict)
				_log('ANALYSIS', f"Added new analysis result for {analysis_key}")
			except Exception as e:
				_log('ANALYSIS', f"Error saving analysis results: {e}")
		

	plt.show(block=True)
	

def plot_result(base_dir, uw_start=30):
    """
    Interactive animation of counts vs position over time.
    - X-axis: position r_arr (cm)
    - Y-axis: X-ray counts
    - Frames: time slices defined by bin_centers (ms)
    """
    from matplotlib.widgets import Slider

    # Load cached results
    analysis_file = os.path.join(base_dir, 'analysis_results.npy')
    tracking_file = os.path.join(base_dir, 'tracking_result.npy')

    analysis_dict = np.load(analysis_file, allow_pickle=True).item()
    tracking_dict = np.load(tracking_file, allow_pickle=True).item()

    # Flatten all points: arrays of time (ms), radius (cm), counts (raw)
    all_t_ms = []
    all_r_cm = []
    all_c = []

    for key, item in tracking_dict.items():
        if item[1] is None:
            # missing ct for this video
            continue
        prefixes = os.path.basename(key)[:2]
        shot_numbers = int(os.path.basename(key).split('_shot')[1][:3])
        t0 = item[1]  # seconds

        analysis_key = f"{prefixes}_{shot_numbers:03d}"
        pulse_tarr, pulse_amp = analysis_dict.get(analysis_key, ([], []))
        if len(pulse_tarr) == 0:
            continue

        # bin_centers in ms; counts are per-bin photon counts
        bin_centers, counts = counts_per_bin(pulse_tarr, pulse_amp, bin_width=1)
        # Convert to seconds for kinematics, then to cm
        time_seconds = (bin_centers + uw_start) * 1e-3
        r_arr_cm = get_pos_freefall(time_seconds, t0) * 100.0

        all_t_ms.extend(bin_centers.tolist())
        all_r_cm.extend(r_arr_cm.tolist())
        all_c.extend(counts.tolist())

    if len(all_t_ms) == 0:
        _log('PLOT', 'No data available to plot')
        return

    all_t_ms = np.asarray(all_t_ms, dtype=float)
    all_r_cm = np.asarray(all_r_cm, dtype=float)
    all_c = np.asarray(all_c, dtype=float)

    # Stable axes across animation
    pos_min, pos_max = float(np.min(all_r_cm)), float(np.max(all_r_cm))
    pad = max(1.0, 0.05 * (pos_max - pos_min))
    y_max = float(np.max(all_c)) if all_c.size else 1.0

    tmin = float(np.min(all_t_ms))
    tmax = float(np.max(all_t_ms))
    # Discrete 1 ms ticks for the slider
    frame_ticks = np.arange(np.floor(tmin), np.ceil(tmax) + 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time (ms)',
                    tmin, tmax,
                    valinit=frame_ticks[0], valstep=frame_ticks)

    def draw_frame(t_ms):
        ax.clear()
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('X-ray Counts')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(pos_min - pad, pos_max + pad)
        ax.set_ylim(0, y_max * 1.1)

        # Select all points within the 1 ms bin centered at t_ms
        mask = (all_t_ms >= (t_ms - 0.5)) & (all_t_ms < (t_ms + 0.5))
        r_vals = all_r_cm[mask]
        c_vals = all_c[mask]

        if r_vals.size > 0:
            # Plot lines between points in radius order for readability
            idx = np.argsort(r_vals)
            ax.plot(r_vals[idx], c_vals[idx], 'o-', linewidth=2, markersize=6, alpha=0.85, color='tab:blue')
            ax.set_title(f'X-ray Counts vs Position  —  t = {t_ms:.1f} ms   (n={r_vals.size})')
        else:
            ax.set_title(f'X-ray Counts vs Position  —  t = {t_ms:.1f} ms   (n=0)')

        fig.canvas.draw_idle()

    def on_slide(val):
        draw_frame(slider.val)

    slider.on_changed(on_slide)

    # Initial frame
    draw_frame(frame_ticks[0])

    # Play/Pause controller
    class Player:
        def __init__(self):
            self.play = False
            self.idx = 0
        def toggle(self, event=None):
            self.play = not self.play
            if self.play:
                # jump to nearest tick
                self.idx = int(np.argmin(np.abs(frame_ticks - slider.val)))
                self.loop()
        def loop(self):
            if not self.play:
                return
            slider.set_val(frame_ticks[self.idx])
            self.idx = (self.idx + 1) % len(frame_ticks)
            fig.canvas.start_event_loop(0.1)
            self.loop()

    player = Player()
    ax_btn = plt.axes([0.85, 0.1, 0.1, 0.05])
    btn = plt.Button(ax_btn, 'Play/Pause')
    btn.on_clicked(player.toggle)

    plt.show(block=True)
# def plot_result(base_dir, uw_start=30):

#     # Load analysis_dict
#     analysis_file = os.path.join(base_dir, 'analysis_results.npy')
#     tracking_file = os.path.join(base_dir, 'tracking_result.npy')

#     analysis_dict = np.load(analysis_file, allow_pickle=True).item()
#     tracking_dict = np.load(tracking_file, allow_pickle=True).item()


#     _log('PLOT', "Creating single combined X-ray counts plot with all data...")
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))

#     all_bin_centers, all_r_positions, all_counts = [], [], []
#     for key, item in tracking_dict.items():
#         if item[1] is None:
#             _log('PLOT', f"Skipping {key} due to missing tracking data.")
#             continue
#         prefixes = os.path.basename(key)[:2]
#         shot_numbers = int(os.path.basename(key).split('_shot')[1][:3])
#         t0 = item[1]

#         analysis_key = f"{prefixes}_{shot_numbers:03d}"
#         pulse_tarr, pulse_amp = analysis_dict.get(analysis_key, ([], []))
#         if len(pulse_tarr) == 0:
#             _log('PLOT', f"Skipping {analysis_key} due to missing analysis data.")
#             continue
#         bin_centers, counts = counts_per_bin(pulse_tarr, pulse_amp, bin_width=1)
#         time_seconds = (bin_centers + uw_start) * 1e-3
#         r_arr = get_pos_freefall(time_seconds, t0) * 100 # convert to cm

#         ax.scatter(bin_centers,r_arr, c=counts, s=50, alpha=0.7, cmap='viridis')
#         print(f"Plotted {prefixes} shot {shot_numbers}")

#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Position (cm)')
#     ax.grid(True)
#     plt.tight_layout()
#     plt.draw()
#     plt.show(block=True)





#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":

	base_dir = r"F:\AUG2025\P24"

	plot_result(base_dir, uw_start=30)

	if False:
		fn = "11_He1kG430G_5800A_K-30_2025-08-12.hdf5"
		xray_wt_cam(base_dir, fn)