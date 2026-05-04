#!/usr/bin/env python3
"""Compare averaged Bdot STFT between two shot groups in P24:
   (A) shots whose tracked y(t) fit reaches y_target (cm) inside the
       microwave-heating window, and
   (B) shots where object tracking failed (no object detected).
"""

import glob
import os
import re

import numpy as np
import h5py
import matplotlib.pyplot as plt

from lapd_io import log, get_bdot_data
from read.read_scope_data import read_hdf5_all_scopes_channels
from process_bdot import calculate_bdot_stft
from plot_bdot import plot_bdot_stft_comparison


# ============================================================
# User-tunable selection parameters
# ============================================================
dir_path = r"E:\AUG2025\P23"
# Pass-y group: keep shots whose linear-fit y(t) lies anywhere in
# Y_RANGE_CM (cm) for some t in the heating window below.
Y_RANGE_CM = (30.0, 40.0)

# Microwave heating window (seconds). Used as the time gate for the
# pass-y selection. Defaults match 'uwave: 30 - 45 ms' from the HDF5
# description; override here to ignore the metadata.
UW_START_S = 0.030
UW_END_S   = 0.045

# Set to True to read the heating window from each HDF5's description
# instead of using the constants above.
USE_UW_FROM_HDF5 = False

# STFT / FFT parameters.
FREQ_BINS = 1000
OVERLAP_FRACTION = 0.05
FREQ_MIN = 50e6
FREQ_MAX = 1000e6

# Number of shots per group for the comparison.
N_PER_GROUP = 5

# Tracking results filename (relative to dir_path).
TRACKING_FILENAME = "tracking_result.npy"

# Where to save the averaged-STFT comparison figure (set to None to skip saving).
SAVE_FIG_PATH = r"C:\Users\hjia9\Documents\lapd\e-ring\diagnostic_fig\compare_bdot_groups.png"

# Toggle which actions run when this file is executed as a script.
RUN_SHOW_EXAMPLE = True
RUN_COMPARE_GROUPS = False
# ============================================================

CINE_PREFIX_RE = re.compile(r"^(\d{2})_.*_shot(\d{3})\.cine$", re.IGNORECASE)
UWAVE_LINE_RE = re.compile(
	r"uwave\s*:\s*([\d.]+)\s*-\s*([\d.]+)\s*ms", re.IGNORECASE
)


def _find_description_text(f):
	for k in f.attrs:
		if k.lower() == "description":
			val = f.attrs[k]
			if isinstance(val, bytes):
				return val.decode("utf-8", errors="replace")
			if isinstance(val, np.ndarray):
				if val.dtype.kind in ("S", "O"):
					return "\n".join(
						v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
						for v in val.tolist()
					)
			return str(val)
	return ""


def parse_uwave_window_s(hdf5_path):
	"""Return (uw_start_s, uw_end_s) parsed from the HDF5 description's
	Timing section (line like 'uwave: 30 - 45 ms')."""
	with h5py.File(hdf5_path, "r") as f:
		desc = _find_description_text(f)
	m = UWAVE_LINE_RE.search(desc)
	if m is None:
		raise ValueError(
			f"No 'uwave: A - B ms' line found in description of {os.path.basename(hdf5_path)}"
		)
	uw_start_ms, uw_end_ms = float(m.group(1)), float(m.group(2))
	return uw_start_ms * 1e-3, uw_end_ms * 1e-3


def _build_prefix_to_hdf5(base_dir):
	mapping = {}
	for path in glob.glob(os.path.join(base_dir, "*.hdf5")):
		prefix = os.path.basename(path)[:2]
		mapping[prefix] = path
	return mapping


def _parse_cine_basename(cine_path):
	"""Return (file_prefix, shot_num) or None if basename doesn't match."""
	m = CINE_PREFIX_RE.match(os.path.basename(cine_path))
	if m is None:
		return None
	return m.group(1), int(m.group(2))


def select_shots(tracking_npy_path, base_dir, n_pass=5, n_fail=5,
				 y_range_cm=Y_RANGE_CM):
	"""Select up to n_pass 'pass-y' shots and n_fail 'fail' shots from
	tracking_result.npy. Pass-y means y(t) is inside y_range_cm for at least
	some t inside the heating window. Returns (pass_map, fail_map, diagnostics).
	"""
	y_lo, y_hi = sorted(y_range_cm)
	tracking_dict = np.load(tracking_npy_path, allow_pickle=True).item()
	prefix_to_hdf5 = _build_prefix_to_hdf5(base_dir)
	uw_window_cache = {}

	pass_candidates = []  # (hdf5_path, shot_num, t_star_s)
	fail_candidates = []  # (hdf5_path, shot_num)

	for cine_path, entry in tracking_dict.items():
		parsed = _parse_cine_basename(cine_path)
		if parsed is None:
			continue
		prefix, shot_num = parsed
		hdf5_path = prefix_to_hdf5.get(prefix)
		if hdf5_path is None:
			continue

		if not isinstance(entry, dict) or "y_slope" not in entry:
			continue

		n_pts = entry.get("n_points", 0)
		y_slope = entry.get("y_slope", float("nan"))
		y_intercept = entry.get("y_intercept", float("nan"))

		if n_pts < 2 or not np.isfinite(y_slope):
			fail_candidates.append((hdf5_path, shot_num))
			continue

		if y_slope == 0:
			continue

		if USE_UW_FROM_HDF5:
			if hdf5_path not in uw_window_cache:
				uw_window_cache[hdf5_path] = parse_uwave_window_s(hdf5_path)
			uw_start_s, uw_end_s = uw_window_cache[hdf5_path]
		else:
			uw_start_s, uw_end_s = UW_START_S, UW_END_S

		# y(t) at the two heating-window edges; trajectory is linear so
		# y inside the window spans [min(y_a, y_b), max(y_a, y_b)].
		y_a = y_intercept + y_slope * uw_start_s
		y_b = y_intercept + y_slope * uw_end_s
		y_traj_lo = min(y_a, y_b)
		y_traj_hi = max(y_a, y_b)

		# Overlap test: trajectory's y-range during heating intersects [y_lo, y_hi].
		if y_traj_hi < y_lo or y_traj_lo > y_hi:
			continue

		# Representative t* = midpoint of overlap, mapped back to t.
		y_overlap_mid = 0.5 * (max(y_traj_lo, y_lo) + min(y_traj_hi, y_hi))
		t_star = (y_overlap_mid - y_intercept) / y_slope

		pass_candidates.append((hdf5_path, shot_num, t_star))

	pass_candidates.sort(key=lambda r: (r[0], r[1]))
	fail_candidates.sort(key=lambda r: (r[0], r[1]))
	pass_sel = pass_candidates[:n_pass]
	fail_sel = fail_candidates[:n_fail]

	pass_map = {}
	for hp, sn, ts in pass_sel:
		pass_map.setdefault(hp, []).append((sn, ts))
	fail_map = {}
	for hp, sn in fail_sel:
		fail_map.setdefault(hp, []).append((sn, None))

	diag = {
		"n_pass_found": len(pass_candidates),
		"n_fail_found": len(fail_candidates),
		"pass_selected": pass_sel,
		"fail_selected": fail_sel,
	}
	return pass_map, fail_map, diag


def compute_group_avg_stft(shot_map):
	"""Compute per-channel averaged Bdot STFT across all shots in shot_map.

	shot_map: {hdf5_path: [(shot_num, _ignored), ...]}
	Returns (avg_stft_matrices, descriptions, stft_tarr, freq_arr).
	"""
	all_matrices = {}
	stft_tarr = None
	freq_arr = None
	descriptions = {}

	for hdf5_path, shot_list in shot_map.items():
		with h5py.File(hdf5_path, "r") as f:
			for shot_num, _ in shot_list:
				log("BDOT", f"{os.path.basename(hdf5_path)} shot {shot_num}")
				result = read_hdf5_all_scopes_channels(f, shot_num)
				tarr_B, bdot_data, descs = get_bdot_data(f, result)
				if tarr_B is None or len(bdot_data) == 0:
					log("BDOT", f"  no Bdot data, skipping")
					continue
				descriptions = descs
				tarr_out, freq_out, stft_matrices = calculate_bdot_stft(
					tarr_B, bdot_data, FREQ_BINS, OVERLAP_FRACTION,
					FREQ_MIN, FREQ_MAX,
				)
				for ch, m in stft_matrices.items():
					if m is None:
						continue
					all_matrices.setdefault(ch, []).append(m)
				if tarr_out is not None and freq_out is not None:
					stft_tarr = tarr_out
					freq_arr = freq_out

	avg = {}
	for ch, mats in all_matrices.items():
		avg[ch] = np.mean(np.array(mats), axis=0)
		log("BDOT", f"Averaged {len(mats)} STFT matrices for channel {ch}")

	return avg, descriptions, stft_tarr, freq_arr


def show_example_shot(shot_map):
	"""Display one raw Bdot trace and its STFT for the last shot in shot_map."""
	hdf5_path = list(shot_map.keys())[-1]
	shot_num, _ = shot_map[hdf5_path][0]
	with h5py.File(hdf5_path, "r") as f:
		result = read_hdf5_all_scopes_channels(f, shot_num)
		tarr_B, bdot_data, descs = get_bdot_data(f, result)

	first_ch = sorted(bdot_data.keys())[0]
	sig = bdot_data[first_ch]
	stft_t, freq, stft_mats = calculate_bdot_stft(
		tarr_B, {first_ch: sig}, FREQ_BINS, OVERLAP_FRACTION,
		FREQ_MIN, FREQ_MAX,
	)
	stft_mat = stft_mats[first_ch]

	import matplotlib.colors as mcolors
	fig, axes = plt.subplots(2, 1, figsize=(8, 7),
							 num=f"Example_{os.path.basename(hdf5_path)}_shot{shot_num}_{first_ch}")
	axes[0].plot(tarr_B * 1e3, sig, lw=0.5)
	axes[0].set_xlabel("Time (ms)")
	axes[0].set_ylabel(f"{first_ch} signal")
	axes[0].set_title(descs.get(first_ch, first_ch))

	safe = stft_mat.copy()
	pos = safe[safe > 0]
	vmin = pos.min() if pos.size else 1e-10
	safe[safe <= 0] = vmin
	im = axes[1].imshow(
		safe.T, aspect="auto", origin="lower",
		extent=[stft_t[0] * 1e3, stft_t[-1] * 1e3, freq[0] / 1e6, freq[-1] / 1e6],
		interpolation="None", cmap="jet",
		norm=mcolors.LogNorm(vmin=vmin, vmax=safe.max()),
	)
	axes[1].set_xlabel("Time (ms)")
	axes[1].set_ylabel("Frequency (MHz)")
	fig.colorbar(im, ax=axes[1], label="Magnitude")
	plt.tight_layout()
	plt.show(block=True)


def run_selection(base_dir=dir_path,
				  tracking_filename=TRACKING_FILENAME,
				  y_range_cm=Y_RANGE_CM,
				  n_per_group=N_PER_GROUP):
	"""Run shot selection and log diagnostics. Returns
	(pass_map, fail_map, diag, tracking_npy_path) or None if too few shots."""
	tracking_npy_path = os.path.join(base_dir, tracking_filename)

	pass_map, fail_map, diag = select_shots(
		tracking_npy_path, base_dir,
		n_pass=n_per_group, n_fail=n_per_group, y_range_cm=y_range_cm,
	)
	log("SEL", f"pass-y candidates total: {diag['n_pass_found']}; "
			   f"fail candidates total: {diag['n_fail_found']}")
	log("SEL", "Pass-y selected (hdf5, shot, t* ms):")
	for hp, sn, ts in diag["pass_selected"]:
		log("SEL", f"  {os.path.basename(hp)}  shot{sn:03d}  t*={ts*1e3:.2f} ms")
	log("SEL", "Tracking-fail selected (hdf5, shot):")
	for hp, sn in diag["fail_selected"]:
		log("SEL", f"  {os.path.basename(hp)}  shot{sn:03d}")

	n_pass_sel = sum(len(v) for v in pass_map.values())
	n_fail_sel = sum(len(v) for v in fail_map.values())
	if n_pass_sel < n_per_group or n_fail_sel < n_per_group:
		log("SEL", f"Aborting: need {n_per_group} per group, "
				   f"got pass={n_pass_sel}, fail={n_fail_sel}")
		return None

	# Sanity check: verify y(t*) for first pass-y shot
	if diag["pass_selected"]:
		hp, sn, ts = diag["pass_selected"][0]
		tracking_dict = np.load(tracking_npy_path, allow_pickle=True).item()
		prefix = os.path.basename(hp)[:2]
		for cp, entry in tracking_dict.items():
			parsed = _parse_cine_basename(cp)
			if parsed and parsed[0] == prefix and parsed[1] == sn:
				y_at = entry["y_intercept"] + entry["y_slope"] * ts
				uw0, uw1 = (parse_uwave_window_s(hp) if USE_UW_FROM_HDF5
							else (UW_START_S, UW_END_S))
				log("CHK", f"Sanity: shot {prefix}_{sn:03d}  y(t*)={y_at:.3f} cm "
						   f"(target range {y_range_cm[0]:g}..{y_range_cm[1]:g}); "
						   f"window=[{uw0*1e3:.1f}, {uw1*1e3:.1f}] ms; "
						   f"t*={ts*1e3:.2f} ms")
				break

	return pass_map, fail_map, diag, tracking_npy_path


def compare_bdot_groups(pass_map, fail_map, base_dir=dir_path):
	"""Compute averaged STFTs for the two groups and produce the comparison plot."""
	log("BDOT", "=== Group A: pass-y ===")
	group_a = compute_group_avg_stft(pass_map)
	log("BDOT", "=== Group B: tracking failed ===")
	group_b = compute_group_avg_stft(fail_map)

	port_match = re.search(r"P\d+", base_dir)
	port_label = port_match.group(0) if port_match else "P??"
	plot_bdot_stft_comparison(
		group_a, group_b,
		labels=(f"with Tungsten {port_label}", "no Tungsten"),
		save_path=SAVE_FIG_PATH,
	)
	return group_a, group_b


if __name__ == "__main__":
	if RUN_SHOW_EXAMPLE or RUN_COMPARE_GROUPS:
		sel = run_selection()
		if sel is not None:
			pass_map, fail_map, _diag, _tracking_path = sel
			if RUN_SHOW_EXAMPLE:
				show_example_shot(pass_map)
			if RUN_COMPARE_GROUPS:
				compare_bdot_groups(pass_map, fail_map)
