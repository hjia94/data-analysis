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
