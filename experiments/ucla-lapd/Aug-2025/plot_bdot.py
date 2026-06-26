#!/usr/bin/env python3
"""Plotting routines for averaged Bdot STFT data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from lapd_io import log

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})


def _floor_for_lognorm(matrix):
	"""Replace non-positive entries with the smallest positive value so the
	matrix is safe for matplotlib LogNorm. Returns (safe_matrix, vmin)."""
	safe = matrix.copy()
	vmin = safe[safe > 0].min() if np.any(safe > 0) else 1e-10
	safe[safe <= 0] = vmin
	return safe, vmin


def plot_averaged_bdot_stft(stft_matrices, description, stft_tarr, freq_arr):
	"""Plot averaged Bdot STFT data for each channel.

	Parameters:
	- stft_matrices: Dictionary of averaged STFT matrices by channel
	- description: Dict mapping channel name to description string
	- stft_tarr: Time array for STFT (seconds)
	- freq_arr: Frequency array for STFT (Hz)
	"""
	num_channels = len(stft_matrices)
	if num_channels == 0:
		log('PLOT', "No STFT matrices to plot")
		return None

	fig, axes = plt.subplots(num_channels, 1, figsize=(8, 8),
							num="Averaged_Bdot_STFT", sharex=True)

	if num_channels == 1:
		axes = [axes]

	channels = sorted(stft_matrices.keys())

	for i, channel in enumerate(channels):
		matrix = stft_matrices[channel]

		positive_matrix, min_positive = _floor_for_lognorm(matrix)

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

	axes[-1].set_xlabel('Time (ms)')
	plt.show(block=True)


def _strip_x100(text):
	"""Remove standalone 'X100' / 'x100' tokens from a description string."""
	import re as _re
	return _re.sub(r"\s*[xX]100\s*", " ", text).strip()


def plot_bdot_stft_comparison(group_a, group_b, labels=("Group A", "Group B"),
							  save_path=None):
	"""Plot averaged Bdot STFT for two groups side-by-side per channel.

	Each group is the tuple returned by compute_group_avg_stft:
	(stft_matrices, descriptions, stft_tarr, freq_arr).
	"""
	stft_a, desc_a, tarr_a, freq_a = group_a
	stft_b, _desc_b, tarr_b, freq_b = group_b

	channels = sorted(set(stft_a.keys()) & set(stft_b.keys()))
	if not channels:
		log('PLOT', "No common channels between groups")
		return None

	num_channels = len(channels)
	fig, axes = plt.subplots(num_channels, 2, figsize=(10, 2.5 * num_channels + 1),
							 num="Bdot_STFT_comparison", sharex=True, sharey=True,
							 squeeze=False)

	# Shared LogNorm across all panels.
	safe_cache = {}
	global_vmin = np.inf
	global_vmax = -np.inf
	for ch in channels:
		safe_a, vmin_a = _floor_for_lognorm(stft_a[ch])
		safe_b, vmin_b = _floor_for_lognorm(stft_b[ch])
		safe_cache[ch] = (safe_a, safe_b)
		global_vmin = min(global_vmin, vmin_a, vmin_b)
		global_vmax = max(global_vmax, safe_a.max(), safe_b.max())
	norm = colors.LogNorm(vmin=global_vmin, vmax=global_vmax)

	im = None
	for i, ch in enumerate(channels):
		safe_a, safe_b = safe_cache[ch]
		for col, (safe, tarr, freq) in enumerate([
			(safe_a, tarr_a, freq_a),
			(safe_b, tarr_b, freq_b),
		]):
			ax = axes[i, col]
			im = ax.imshow(safe.T, aspect='auto', origin='lower',
						   extent=[tarr[0]*1e3, tarr[-1]*1e3,
								   freq[0]/1e6, freq[-1]/1e6],
						   interpolation='None', cmap='jet', norm=norm)
			if i == 0:
				ax.set_title(labels[col])
			if col == 0:
				ax.set_ylabel(f"{_strip_x100(desc_a.get(ch, ch))}\nFreq (MHz)")
			if i == num_channels - 1:
				ax.set_xlabel('Time (ms)')

	fig.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.08,
						wspace=0.05, hspace=0.05)
	cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.87])
	fig.colorbar(im, cax=cbar_ax, label='Magnitude')
	if save_path:
		import os as _os
		_os.makedirs(_os.path.dirname(save_path), exist_ok=True)
		fig.savefig(save_path, dpi=150, bbox_inches='tight')
		log('PLOT', f"Saved comparison figure to {save_path}")

