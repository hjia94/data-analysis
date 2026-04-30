#!/usr/bin/env python3
"""Process Bdot scope data from LAPD HDF5 files: per-shot STFT and shot-averaged STFT."""

import os

import numpy as np
import h5py

from lapd_io import log, get_bdot_data
from read.read_scope_data import read_hdf5_all_scopes_channels
from data_analysis_utils import calculate_stft
from plot_bdot import plot_averaged_bdot_stft


def calculate_bdot_stft(tarr, bdot_data, freq_bins=1000, overlap_fraction=0.05, freq_min=200e6, freq_max=2000e6):
	"""Calculates STFT for each Bdot signal in bdot_data.

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
	"""
	stft_matrices = {}
	stft_time = None
	freq = None

	for channel, data in bdot_data.items():
		if data is not None:
			freq, stft_matrix, stft_time = calculate_stft(tarr, data, freq_bins, overlap_fraction, 'hanning', freq_min, freq_max)
			stft_matrices[channel] = stft_matrix
			log('STFT', f"Calculated STFT for channel {channel}")
		else:
			stft_matrices[channel] = None
			log('STFT', f"Skipped STFT for channel {channel} (no data)")

	return stft_time, freq, stft_matrices


def process_bdot(ifn, freq_bins=1000, overlap_fraction=0.05, freq_min=50e6, freq_max=1000e6, plot=True):
	"""Process Bdot data from an HDF5 file by averaging STFT across all shots for each channel.

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
	log('BDOT', f"Processing Bdot data from {os.path.basename(ifn)}")

	with h5py.File(ifn, 'r') as f:
		shot_numbers = f['Control/FastCam']['shot number'][()]
		log('BDOT', f"Found {len(shot_numbers)} shots")

		all_stft_matrices = {}  # Format: {channel: [stft_matrix1, stft_matrix2, ...]}
		stft_tarr_final = None
		freq_arr_final = None
		descriptions = {}

		for shot_num in shot_numbers:
			log('BDOT', f"Processing shot {shot_num}")

			result = read_hdf5_all_scopes_channels(f, shot_num)
			tarr_B, bdot_data, descriptions = get_bdot_data(f, result)

			if tarr_B is None or len(bdot_data) == 0:
				log('BDOT', f"No Bdot data for shot {shot_num}")
				continue

			stft_tarr, freq_arr, stft_matrices = calculate_bdot_stft(tarr_B, bdot_data, freq_bins, overlap_fraction, freq_min, freq_max)

			for channel, matrix in stft_matrices.items():
				if channel not in all_stft_matrices:
					all_stft_matrices[channel] = []
				if matrix is not None:
					all_stft_matrices[channel].append(matrix)

			if stft_tarr is not None and freq_arr is not None:
				stft_tarr_final = stft_tarr
				freq_arr_final = freq_arr

	avg_stft_matrices = {}
	for channel, matrices in all_stft_matrices.items():
		if matrices:
			avg_stft_matrices[channel] = np.mean(np.array(matrices), axis=0)
			log('BDOT', f"Averaged {len(matrices)} STFT matrices for channel {channel}")

	if plot and avg_stft_matrices:
		plot_averaged_bdot_stft(avg_stft_matrices, descriptions, stft_tarr_final, freq_arr_final)

	return avg_stft_matrices, descriptions, stft_tarr_final, freq_arr_final


if __name__ == "__main__":
	ifn = r"F:\AUG2025\P24\27_He1kG430G_5800A_K-5_2025-08-12.hdf5"
	process_bdot(ifn)
