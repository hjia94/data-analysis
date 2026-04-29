"""
Generate and save tracking results for a folder of cines.

Workflow
--------
1. ``track_shots(base_dir)`` walks the cine files in ``base_dir`` and runs
   ``track_object_sparse`` on each, sampling ~5 well-separated frames and
   saving a line fit for the trajectory to ``base_dir/tracking_result.npy``.
   The per-port ``calibration_factor_P{N}.npy`` is loaded once at the top so
   the saved fits are in cm directly.
2. ``verify_tracking(base_dir)`` loads that npy and overlays each shot's
   sample points and saved line, plus a ``y = 0`` reference and the
   derived centre-crossing time ``ct``. Used to confirm visually that the
   sparse fit lies through the points and crosses zero where expected.

On-disk schema for tracking_result.npy
--------------------------------------
``{cine_path: <dict returned by track_object_sparse>}``. Each entry stores:
``x_slope, x_intercept, y_slope, y_intercept`` (cm vs s, chamber-centre
frame), ``cm_per_px`` (the value used at fit time), ``n_points``,
``t_min``, ``t_max``, ``ct = -y_intercept / y_slope``, and
``sample_points`` ((n_points, 3) int — frame_idx, rel_x_px, rel_y_px).
Failed shots have NaN coefficients and ``n_points = 0``.
"""

import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np

from read_cine import read_cine, convert_cine_to_avi
from track_object import track_object_sparse


def _resolve_calibration(base_dir, calibration_file):
    if calibration_file is None:
        m = re.search(r"P\d+", base_dir)
        if m is None:
            raise ValueError(
                f"Could not parse port (P\\d+) from base_dir={base_dir!r}; "
                "pass calibration_file explicitly."
            )
        calibration_file = rf"E:\calibration_factor_{m.group(0)}.npy"
    calib = np.load(calibration_file, allow_pickle=True).item()
    return float(calib["cm_per_px_mean"])


def track_shots(base_dir, pattern="*.cine", filenames=None, overwrite=False,
                calibration_file=None):
    """Track each cine in ``base_dir`` and append to ``tracking_result.npy``.

    Args:
        base_dir: Folder containing cine files (and where the npy is written).
        pattern: Glob pattern used when ``filenames`` is None.
        filenames: Optional explicit list of cine basenames or full paths to
            track. Overrides ``pattern`` when provided.
        overwrite: When True, re-track and replace any existing entry for the
            same cine path. When False, skip entries already present.
        calibration_file: Path to ``calibration_factor_P{N}.npy``. Defaults
            to ``E:\\calibration_factor_{port}.npy`` where ``port`` is parsed
            from ``base_dir``.
    """
    cm_per_px = _resolve_calibration(base_dir, calibration_file)

    if filenames is not None:
        cine_paths = [
            fn if os.path.isabs(fn) else os.path.join(base_dir, fn)
            for fn in filenames
        ]
    else:
        cine_paths = sorted(glob.glob(os.path.join(base_dir, pattern)))

    if not cine_paths:
        print(f"[track_shots] no cines matching {pattern} in {base_dir}")
        return

    out_path = os.path.join(base_dir, "tracking_result.npy")
    if os.path.exists(out_path):
        tracking_dict = np.load(out_path, allow_pickle=True).item()
    else:
        tracking_dict = {}

    def _persist():
        try:
            np.save(out_path, tracking_dict)
        except Exception as e:
            print(f"[track_shots] failed to save {out_path}: {e}")

    total = len(cine_paths)
    processed = 0
    elapsed_total = 0.0
    try:
        for i, cine_path in enumerate(cine_paths, start=1):
            name = os.path.basename(cine_path)
            if cine_path in tracking_dict and not overwrite:
                print(f"[track_shots] skip {name} (already cached)")
                continue
            if not os.path.exists(cine_path):
                print(f"[track_shots] missing {cine_path}; skipping")
                continue

            avi_path = os.path.splitext(cine_path)[0] + ".avi"
            if not os.path.exists(avi_path):
                # Cine is only decoded when the avi sidecar doesn't yet exist;
                # the sparse tracker reads frames straight from the avi after.
                print(f"[track_shots] converting {name} -> .avi")
                try:
                    _, frarr, _ = read_cine(cine_path)
                except Exception as e:
                    print(f"[track_shots] {name}: read_cine failed: {e}")
                    continue
                convert_cine_to_avi(frarr, avi_path)

            print(f"[track_shots] tracking {name} ({i}/{total})")
            t0 = time.monotonic()
            entry = track_object_sparse(avi_path, cine_path, cm_per_px=cm_per_px)
            dt = time.monotonic() - t0
            processed += 1
            elapsed_total += dt
            avg = elapsed_total / processed
            remaining = total - i
            eta_s = avg * remaining
            print(f"[track_shots] {name}: {dt:.1f}s (avg {avg:.1f}s/shot, "
                  f"{remaining} left, ETA {eta_s/60:.1f} min)")

            n = entry["n_points"]
            if n == 0:
                print(f"[track_shots] {name}: no object tracked")
            elif n < 5:
                print(f"[track_shots] {name}: only {n} sample points")

            # Persist immediately after each successful track so a later
            # interrupt or crash doesn't lose the completed shot.
            tracking_dict[cine_path] = entry
            _persist()
    except KeyboardInterrupt:
        print("[track_shots] KeyboardInterrupt — saving progress and exiting")
        _persist()
        raise

    print(f"[track_shots] wrote {out_path} ({len(tracking_dict)} entries)")


def count_y_passes(base_dir, y, t_start, t_stop, tracking_dict=None):
    """Count tracked shots whose fitted trajectory crosses ``y`` in a time window.

    Args:
        base_dir: Folder containing ``tracking_result.npy``. Ignored when
            ``tracking_dict`` is provided.
        y: Chamber-centred vertical position in cm, or 1-D array of positions
            (e.g. ``r_arr_cm`` derived from ``bin_centers``).
        t_start: Inclusive start of the time window in ms, or 1-D array
            matching ``y`` (e.g. ``bin_centers - half_width``).
        t_stop: Inclusive end of the time window in ms, or 1-D array matching
            ``y`` (e.g. ``bin_centers + half_width``).
        tracking_dict: Optional preloaded tracking dictionary. Pass this when
            calling repeatedly to avoid reloading the file from disk each time.

    Returns:
        int when ``y`` is scalar; 1-D int array of per-bin counts when ``y``
        is array-like. Each element counts how many valid tracking entries have
        a fitted crossing time inside ``[t_start[i], t_stop[i]]`` at ``y[i]``.
    """
    scalar_input = np.ndim(y) == 0
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    t_start_arr = np.atleast_1d(np.asarray(t_start, dtype=float))
    t_stop_arr = np.atleast_1d(np.asarray(t_stop, dtype=float))

    if np.any(t_start_arr > t_stop_arr):
        raise ValueError("t_start must be less than or equal to t_stop")

    if tracking_dict is None:
        out_path = os.path.join(base_dir, "tracking_result.npy")
        tracking_dict = np.load(out_path, allow_pickle=True).item()

    counts = np.zeros(len(y_arr), dtype=int)
    for entry in tracking_dict.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("n_points", 0) < 2:
            continue

        y_slope = entry.get("y_slope", float("nan"))
        y_intercept = entry.get("y_intercept", float("nan"))
        if not np.isfinite(y_slope) or not np.isfinite(y_intercept) or y_slope == 0:
            continue

        t_cross_ms = (y_arr - y_intercept) / y_slope * 1e3
        counts += (t_start_arr <= t_cross_ms) & (t_cross_ms <= t_stop_arr)

    return int(counts[0]) if scalar_input else counts


def verify_tracking(base_dir, show=True):
    """Plot a histogram of centre-crossing times (ct) over all tracked shots.

    Args:
        base_dir: Folder containing ``tracking_result.npy``.
        show: When True, blocks on ``plt.show``. Set False for non-interactive
            scripts; the caller is then responsible for the figure.
    """
    out_path = os.path.join(base_dir, "tracking_result.npy")
    tracking_dict = np.load(out_path, allow_pickle=True).item()

    ct_ms = []
    t30_ms = []
    for entry in tracking_dict.values():
        if not isinstance(entry, dict):
            continue
        y_slope = entry.get("y_slope", float("nan"))
        y_intercept = entry.get("y_intercept", float("nan"))
        ct = entry.get("ct", float("nan"))
        if np.isfinite(ct):
            ct_ms.append(ct * 1e3)
        if np.isfinite(y_slope) and np.isfinite(y_intercept) and y_slope != 0:
            t30_ms.append((30.0 - y_intercept) / y_slope * 1e3)

    if not ct_ms and not t30_ms:
        print(f"[verify_tracking] no finite ct values in {out_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(np.array(ct_ms), bins="auto", edgecolor="k", linewidth=0.5)
    axes[0].set_xlabel("ct (ms)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"y = 0 crossing ({len(ct_ms)} shots)")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(np.array(t30_ms), bins="auto", edgecolor="k", linewidth=0.5)
    axes[1].set_xlabel("t (ms)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"y = +30 cm crossing ({len(t30_ms)} shots)")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(base_dir)
    fig.tight_layout()
    if show:
        plt.show(block=True)


if __name__ == "__main__":
    base_dir = r"E:\good_data\kapton\He3kA_B380G800G_pl0t20_uw15t35"
    track_shots(base_dir, calibration_file=r"E:\calibration_factor_P30.npy")
    verify_tracking(base_dir)
