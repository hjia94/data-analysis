#!/usr/bin/env python3
"""Plotting routines for averaged Bdot STFT data."""

import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from data_analysis.io.scope_reader import read_hdf5_all_scopes_channels
from data_analysis.viz.plot_utils import floor_for_lognorm, plot_stft
from lapd_io import log, get_bdot_data
from process_bdot import calculate_bdot_stft

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})


def _save_figure(fig, save_path, what):
	"""Save fig to save_path, creating its directory. No-op if save_path is falsy.

	Not viz.plot_utils.finalize_figure: that one closes/shows the figure, but
	callers here leave it open for a later plt.show(block=True).
	"""
	if not save_path:
		return
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	fig.savefig(save_path, dpi=150, bbox_inches='tight')
	log('PLOT', f"Saved {what} figure to {save_path}")


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
		plot_stft(stft_tarr, freq_arr, stft_matrices[channel],
				  ax=axes[i], fig=fig, title=description[channel])

	axes[-1].set_xlabel('Time (ms)')
	plt.show(block=True)


def show_example_shot(shot_map, freq_bins=1000, overlap_fraction=0.05,
					  freq_min=50e6, freq_max=1000e6):
	"""Raw Bdot trace over its STFT, for the first channel of the last shot in
	shot_map. STFT defaults match process_bdot.process_bdot."""
	hdf5_path = list(shot_map.keys())[-1]
	shot_num, _ = shot_map[hdf5_path][0]
	with h5py.File(hdf5_path, "r") as f:
		result = read_hdf5_all_scopes_channels(f, shot_num)
		tarr_B, bdot_data, descs = get_bdot_data(f, result)

	first_ch = sorted(bdot_data.keys())[0]
	sig = bdot_data[first_ch]
	stft_t, freq, stft_mats = calculate_bdot_stft(
		tarr_B, {first_ch: sig}, freq_bins, overlap_fraction,
		freq_min, freq_max,
	)

	fig, axes = plt.subplots(2, 1, figsize=(8, 7),
							 num=f"Example_{os.path.basename(hdf5_path)}_shot{shot_num}_{first_ch}")
	axes[0].plot(tarr_B * 1e3, sig, lw=0.5)
	axes[0].set_xlabel("Time (ms)")
	axes[0].set_ylabel(f"{first_ch} signal")
	axes[0].set_title(descs.get(first_ch, first_ch))

	plot_stft(stft_t, freq, stft_mats[first_ch], ax=axes[1], fig=fig)
	axes[1].set_xlabel("Time (ms)")
	plt.tight_layout()
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
		safe_a, vmin_a = floor_for_lognorm(stft_a[ch])
		safe_b, vmin_b = floor_for_lognorm(stft_b[ch])
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
				ax.set_ylabel("Freq (MHz)")
			if i == num_channels - 1:
				ax.set_xlabel('Time (ms)')

	fig.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.08,
						wspace=0.05, hspace=0.05)
	cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.87])
	fig.colorbar(im, cax=cbar_ax, label='Magnitude')
	_save_figure(fig, save_path, "comparison")


def plot_band_power_comparison(power_a, power_b, labels=("Group A", "Group B"),
							   title=None, save_path=None):
	"""Plot shot-averaged band power vs time for two groups, one panel per channel.

	Shaded band is +/- std/sqrt(n), the error on the mean: where the two shaded
	bands overlap, the groups are not distinguishable in that bin.

	power_*: {ch: (bin_centers_s, mean_dB, sem_dB, n_shots)} from
	compute_group_avg_stft.
	"""
	channels = sorted(set(power_a.keys()) & set(power_b.keys()))
	if not channels:
		log('PLOT', "No common channels between groups")
		return None

	num_channels = len(channels)
	fig, axes = plt.subplots(num_channels, 1,
							 figsize=(9, 2.6 * num_channels + 1),
							 num="Bdot_band_power_comparison",
							 sharex=True, squeeze=False)
	axes = axes[:, 0]

	for i, ch in enumerate(channels):
		ax = axes[i]
		for (t_s, p_db, sem_db, n), label, color in [
			(power_a[ch], labels[0], 'tab:red'),
			(power_b[ch], labels[1], 'tab:blue'),
		]:
			t_ms = np.asarray(t_s) * 1e3
			ax.fill_between(t_ms, p_db - sem_db, p_db + sem_db,
							color=color, alpha=0.2, lw=0)
			ax.plot(t_ms, p_db, '-o', ms=5, lw=1.5,
					color=color, label=f"{label}")
		ax.set_ylabel(f"Whistler Wave Power (dB)")
		ax.grid(True, alpha=0.3)
		if i == 0:
			ax.legend(fontsize=14)

	axes[-1].set_xlabel('Time (ms)')
	if title:
		axes[0].set_title(title, fontsize=16)

	fig.tight_layout()
	_save_figure(fig, save_path, "band-power")

