#!/usr/bin/env python3
"""Process X-ray scope data from LAPD HDF5 files: per-shot pulse detection,
persisted to ``analysis_results.npy``."""

import os

import numpy as np
import h5py

from lapd_io import log, get_xray_data
from read.read_scope_data import read_hdf5_all_scopes_channels
from data_analysis_utils import Photons
from tracking_utils import analysis_key


def process_shot_xray(tarr_x, xray_data, min_ts, d, threshold, debug=False):
	"""Process a single shot of xray data and return data for averaging."""
	detector = Photons(tarr_x, xray_data, min_timescale=min_ts, distance_mult=d, tsh_mult=threshold, debug=debug)
	detector.reduce_pulses()
	return detector.pulse_times, detector.pulse_amplitudes


def xray_wt_cam(base_dir, fn):
	"""Run x-ray pulse detection for every shot in an HDF5 file and append
	the per-shot ``(pulse_tarr, pulse_amp)`` to ``analysis_results.npy``.
	"""
	ifn = os.path.join(base_dir, fn)

	# Two-digit prefix from filename (e.g. "02" from "02_He1kG430G_..._2025-08-12.hdf5")
	file_prefix = fn[:2]

	analysis_file = os.path.join(base_dir, 'analysis_results.npy')
	if os.path.exists(analysis_file):
		analysis_dict = np.load(analysis_file, allow_pickle=True).item()
	else:
		analysis_dict = {}

	threshold = [10, 80] # P23/P24 [5,70]; P21 [10, 80]
	min_ts = 0.8e-6
	d = 0.1

	with h5py.File(ifn, 'r') as f:
		log('FILE', f"Opened file {ifn} for processing.")
		shot_numbers = f['Control/FastCam']['shot number'][()]

		for shot_num in shot_numbers:
			key = analysis_key(file_prefix, shot_num)
			if key in analysis_dict:
				log('ANALYSIS', f"Analysis result exists for {key}")
				continue

			log('SHOT', f"Processing shot {shot_num}")
			result = read_hdf5_all_scopes_channels(f, shot_num, include_tarr=True)
			tarr_x, xray_data = get_xray_data(result)

			pulse_tarr, pulse_amp = process_shot_xray(tarr_x, xray_data, min_ts, d, threshold)
			analysis_dict[key] = (pulse_tarr, pulse_amp)
			log('ANALYSIS', f"Added new analysis result for {key}")

	try:
		np.save(analysis_file, analysis_dict)
	except Exception as e:
		log('ANALYSIS', f"Error saving analysis results: {e}")


def batch_process_xray(base_dir):
	"""Batch process all HDF5 files in base_dir using xray_wt_cam,
	skipping files whose prefix already exists in analysis_results.npy."""
	hdf5_files = [f for f in os.listdir(base_dir) if f.endswith('.hdf5')]

	for fn in hdf5_files:
		log('FILE', f"Processing {fn} ...")
		xray_wt_cam(base_dir, fn)


if __name__ == "__main__":
	base_dir = r"E:\AUG2025\P21"
	batch_process_xray(base_dir)
