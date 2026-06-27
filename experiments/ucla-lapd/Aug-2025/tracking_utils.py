#!/usr/bin/env python3
"""Shared helpers for the Aug-2025 LAPD tracking + scope pipeline.

Single source of truth for:
- analysis-key string format ``{file_prefix}_{shot_num:03d}`` (and its inverse,
  parsing a cine basename),
- locating a tracking-dict cine path from (prefix, shot_num),
- validating tracking entries against the sparse-fit schema written by
  ``object_tracking/generate_tracking.py``,
- the scope-frame (ms) -> chamber-frame (s) time conversion,
- evaluating the per-shot linear y-fit at scope times.

Pure Python: no matplotlib, no h5py imports here.
"""

import os

import numpy as np


SCOPE_TO_CHAMBER_S = 1e-3  # multiplier in (t_ms + uw_start_ms) * SCOPE_TO_CHAMBER_S


def analysis_key(file_prefix, shot_num):
    """Canonical analysis-result key: '{file_prefix}_{shot_num:03d}'."""
    return f"{file_prefix}_{int(shot_num):03d}"


def analysis_key_for_basename(basename):
    """Build the analysis key from a cine basename like
    ``02_..._shot017.cine``."""
    return f"{basename[:2]}_{int(basename.split('_shot')[1][:3]):03d}"


def find_cine_path_for_shot(tracking_dict, file_prefix, shot_num):
    """Inverse of ``analysis_key_for_basename``: return the full cine path in
    ``tracking_dict`` whose basename matches ``{prefix}_..._shot{NNN}.cine``,
    or ``None`` if absent."""
    target = f"_shot{int(shot_num):03d}"
    for cine_path in tracking_dict:
        base = os.path.basename(cine_path)
        if base[:2] == file_prefix and target in base:
            return cine_path
    return None


def is_valid_tracking_entry(entry, *, strict_schema=False):
    """Return True iff ``entry`` carries a usable sparse-fit line.

    With ``strict_schema=True`` (movie_maker style), legacy cache entries —
    anything that is not a dict with a ``y_slope`` key — raise ``TypeError``
    to force a re-run of ``object_tracking/generate_tracking.py``. With
    ``strict_schema=False`` (export_xray_npz style), legacy/missing entries
    quietly return False.
    """
    if entry is None:
        return False
    if not isinstance(entry, dict) or "y_slope" not in entry:
        if strict_schema:
            raise TypeError(
                "legacy cache entry; re-run "
                "object_tracking/generate_tracking.py to rebuild "
                "tracking_result.npy with the sparse-fit schema."
            )
        return False
    if entry.get("n_points", 0) < 2:
        return False
    if not np.isfinite(entry.get("y_slope", float("nan"))):
        return False
    if not np.isfinite(entry.get("y_intercept", float("nan"))):
        return False
    return True


def iter_valid_tracking(tracking_dict, *, strict_schema=False):
    """Yield ``(cine_path, entry)`` for every tracking-dict entry that passes
    ``is_valid_tracking_entry``. With ``strict_schema=True``, legacy entries
    raise; without, they are silently skipped."""
    for cine_path, entry in tracking_dict.items():
        try:
            valid = is_valid_tracking_entry(entry, strict_schema=strict_schema)
        except TypeError as exc:
            raise TypeError(f"{cine_path}: {exc}") from exc
        if valid:
            yield cine_path, entry


def scope_ms_to_chamber_s(t_ms, uw_start_ms):
    """Convert scope-trigger-frame time (ms) to chamber-frame time (s).

    The chamber-frame time is what the tracking line fit is parameterized by
    (see ``object_tracking/generate_tracking.py``), so any consumer that
    evaluates the fit at scope times must go through this conversion.
    """
    return (np.asarray(t_ms, dtype=float) + uw_start_ms) * SCOPE_TO_CHAMBER_S


def evaluate_y_cm(entry, t_ms, uw_start_ms):
    """Return chamber-centred y (cm) at scope-frame times ``t_ms``.

    Always returns chamber-centred y. Callers that want "r below chamber
    centre" negate the result explicitly so the sign convention stays
    visible at the call site. Returns a NaN-filled array (same shape as
    ``t_ms``) when the entry is missing or invalid (non-strict mode); use
    ``is_valid_tracking_entry(..., strict_schema=True)`` upstream if you
    want missing-fit shots to raise instead.
    """
    t_arr = np.asarray(t_ms, dtype=float)
    if not is_valid_tracking_entry(entry):
        return np.full_like(t_arr, np.nan, dtype=float)
    t_s = scope_ms_to_chamber_s(t_arr, uw_start_ms)
    return entry["y_intercept"] + entry["y_slope"] * t_s
