#!/usr/bin/env python3
"""Shared HDF5 readers and logger for Aug-2025 LAPD scope data.

Importing this module also configures sys.path so the project's
``read`` and ``data_analysis_utils`` modules are available.
"""

import os
import re
import sys

from scipy import ndimage

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__ if '__file__' in globals() else os.getcwd()), '../..'))
sys.path = [repo_root, f"{repo_root}/read"] + sys.path

from read.read_scope_data import read_scope_channel_descriptions  # noqa: E402


def log(tag, msg):
	"""Prefixed terminal logger used across the Aug-2025 scripts."""
	print(f"[{tag}] {msg}")


def get_magnetron_power_data(f, result, scope_name='magscope'):
	"""Calculate magnetron power from HDF5 file data."""
	if scope_name not in result:
		log('POWER', f"Scope '{scope_name}' not found.")
		return None, None

	tarr = result[scope_name].get('time_array')
	chan_data = result[scope_name].get('channels', {})
	descriptions = read_scope_channel_descriptions(f, scope_name)

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
		log('POWER', "Magnetron power calculated")
	else:
		log('POWER', "Cannot calculate power: missing current or voltage data")

	return tarr, P_data


def get_xray_data(result, scope_name='xrayscope'):
	tarr_x = result[scope_name].get('time_array')
	xray_data = result[scope_name]['channels']['C2']
	return tarr_x, xray_data


def get_bdot_data(f, result, scope_name='bdotscope'):
	tarr = result[scope_name].get('time_array')
	chan_data = result[scope_name].get('channels', {})
	descriptions = read_scope_channel_descriptions(f, scope_name)
	return tarr, chan_data, descriptions
