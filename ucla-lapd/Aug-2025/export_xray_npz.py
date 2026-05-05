#!/usr/bin/env python3
"""Export filtered, baseline-subtracted x-ray data per shot to .npz bundles
for an external collaborator. Joins each shot's x-ray scope trace with the
tungsten-ball y-position derived from the linear trajectory fit in
``tracking_result.npy``.

Per-shot output schema (``{file_prefix}_{shot_num:03d}.npz``):
    tarr_ds_ms           (N,) f8  downsampled scope time, ms, scope-trigger frame
    xray_filt_baseline   (N,) f8  Savitzky-Golay-filtered, downsampled, baseline-subtracted
    y_cm                 (N,) f8  tungsten ball y in cm, chamber-centred (NaN if no tracking)
    file_prefix          scalar   e.g. "02"
    shot_num             scalar
    uw_start_ms          scalar   t_s = (tarr_ds_ms + uw_start_ms) * 1e-3
    has_tracking         scalar bool
    cine_basename        scalar   "" if no tracking
    y_slope, y_intercept scalar   cm vs s, chamber frame (NaN if no tracking)
    cm_per_px            scalar   (NaN if no tracking)
"""

import os
import glob

import numpy as np
import h5py

from lapd_io import log, get_xray_data
from read.read_scope_data import read_hdf5_all_scopes_channels
from data_analysis_utils import Photons
from tracking_utils import (
	evaluate_y_cm,
	find_cine_path_for_shot,
	is_valid_tracking_entry,
)


def export_shot_xray_npz(f_h5, shot_num, file_prefix, tracking_dict,
						 uw_start_ms, out_dir,
						 min_ts=0.8e-6, distance_mult=0.1,
						 threshold=(10, 80), overwrite=False):
	"""Build one shot's npz bundle and write it to ``out_dir``."""
	out_path = os.path.join(out_dir, f"{file_prefix}_{shot_num:03d}.npz")
	if os.path.exists(out_path) and not overwrite:
		log('EXPORT', f"skip {os.path.basename(out_path)} (exists)")
		return

	result = read_hdf5_all_scopes_channels(f_h5, shot_num, include_tarr=True)
	tarr_x, xray_data = get_xray_data(result)

	detector = Photons(tarr_x, xray_data,
					   min_timescale=min_ts,
					   tsh_mult=list(threshold),
					   distance_mult=distance_mult)
	tarr_ds_ms = np.asarray(detector.tarr_ds, dtype=float)
	xray_ds = np.asarray(detector.baseline_subtracted, dtype=float)

	cine_path = find_cine_path_for_shot(tracking_dict, file_prefix, shot_num)
	entry = tracking_dict.get(cine_path) if cine_path else None
	y_cm = evaluate_y_cm(entry, tarr_ds_ms, uw_start_ms)
	has_tracking = is_valid_tracking_entry(entry)

	np.savez_compressed(
		out_path,
		tarr_ds_ms=tarr_ds_ms,
		xray_filt_baseline=xray_ds,
		y_cm=y_cm,
		file_prefix=np.array(file_prefix),
		shot_num=np.int32(shot_num),
		uw_start_ms=np.float64(uw_start_ms),
		has_tracking=np.bool_(has_tracking),
		cine_basename=np.array(os.path.basename(cine_path) if cine_path else ""),
		y_slope=np.float64(entry["y_slope"] if has_tracking else np.nan),
		y_intercept=np.float64(entry["y_intercept"] if has_tracking else np.nan),
		cm_per_px=np.float64(entry["cm_per_px"] if has_tracking and "cm_per_px" in entry else np.nan),
	)
	log('EXPORT', f"wrote {os.path.basename(out_path)} (has_tracking={has_tracking})")


def batch_export_xray_npz(base_dir, tracking_path, uw_start_ms,
						  out_dir=None, overwrite=False):
	"""Export every shot in every HDF5 in ``base_dir`` to one npz per shot.

	Args:
		base_dir: Folder containing ``*.hdf5`` scope files.
		tracking_path: Path to ``tracking_result.npy`` (cine-keyed dict).
		uw_start_ms: Offset (ms) so that
			``t_s = (tarr_ds_ms + uw_start_ms) * 1e-3`` is the
			chamber-frame time used by the tracking fit. Required so this
			never silently misaligns across campaigns.
		out_dir: Defaults to ``{base_dir}/xray_export``.
		overwrite: When True, re-export shots whose npz already exists.
	"""
	if out_dir is None:
		out_dir = os.path.join(base_dir, "xray_export")
	os.makedirs(out_dir, exist_ok=True)

	tracking_dict = np.load(tracking_path, allow_pickle=True).item()
	log('EXPORT', f"loaded {len(tracking_dict)} tracking entries from {tracking_path}")

	hdf5_files = sorted(glob.glob(os.path.join(base_dir, "*.hdf5")))
	for ifn in hdf5_files:
		fn = os.path.basename(ifn)
		file_prefix = fn[:2]
		log('FILE', f"Processing {fn} ...")
		with h5py.File(ifn, 'r') as f:
			shot_numbers = f['Control/FastCam']['shot number'][()]
			for shot_num in shot_numbers:
				try:
					export_shot_xray_npz(
						f, int(shot_num), file_prefix, tracking_dict,
						uw_start_ms=uw_start_ms, out_dir=out_dir,
						overwrite=overwrite,
					)
				except Exception as e:
					log('EXPORT', f"shot {file_prefix}_{int(shot_num):03d} failed: {e}")

	log('EXPORT', f"done; outputs in {out_dir}")


if __name__ == "__main__":
	base_dir = r"E:\AUG2025\P23"
	tracking_path = r"E:\AUG2025\P23\tracking_result.npy"
	# uw_start_ms must be set per-campaign before running.
	batch_export_xray_npz(base_dir, tracking_path, uw_start_ms=15.0)
