'''
General utility functions used across data analysis: file discovery and .npy I/O.

Signal-processing helpers live in ``data_analysis.signal.core`` and the
photon-pulse classes in ``data_analysis.plasma.photons`` -- import them from
there directly.

Author: Jia Han
Date: 2024-02-28

Included functions:
- get_files_in_folder
- get_number_before_keyword
- run_num_of
- find_run_hdf5
- save_to_npy
- read_from_npy
- merge_save_npz
- pack_npz_entries
- pack_nested_npz
- npz_settings_mismatch
- prepare_incremental_npz
- fmt_num
'''

import glob
import os
import re
import numpy as np
from datetime import datetime


def get_files_in_folder(folder_path, modified_date=None, omit_keyword=None):
    """
    Get a list of all files in a given folder and its subfolders.

    Args:
        folder_path (str): The path to the folder.
        modified_date (str, optional): The specific date in the format 'YYYY-MM-DD'. Defaults to None.
        omit_keyword (str, optional): The keyword to omit files. Defaults to None.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if omit_keyword is not None and omit_keyword in file:
                continue
            if modified_date is None:
                file_list.append(file_path)
            else:
                last_modified = os.path.getmtime(file_path)
                last_modified_date = datetime.fromtimestamp(last_modified).date()
                if last_modified_date == datetime.strptime(modified_date, '%Y-%m-%d').date():
                    file_list.append(file_path)
    return file_list


def get_number_before_keyword(string, keyword, verbose=False):
    """
    Extracts the number before a given keyword in a string.

    Parameters:
    string (str): The input string.
    keyword (str): The keyword to search for.
    verbose (bool, optional): If True, prints the number found. Defaults to False.

    Returns:
    float or None: The number found before the keyword, or None if no match is found.
    """
    match = re.search(r'(\d+)'+keyword, string)
    if match:
        if verbose:
            print("Number found:", match.group(1))
        return float(match.group(1))
    else:
        if verbose:
            print("No match found.")
        return None


def run_num_of(ifn):
    """The leading run-number token of a run file (``"02-He-..."`` -> ``"02"``).

    The one home of the ``NN-description.hdf5`` filename convention used by the
    LAPD campaigns -- experiment scripts import this rather than re-splitting
    the basename.
    """
    return os.path.basename(ifn).split("-")[0]


def save_to_npy(data, npy_file_path):
    """
    Save data to a .npy file.

    If the file exists, blocks on ``input()`` for y/n; only 'n' cancels -- any
    other reply (including a typo) overwrites. Not for unattended/batch use.

    Parameters:
    data (numpy.ndarray): The data to be saved.
    npy_file_path (str): The file path to save the .npy file.

    Returns:
    None
    """
    if os.path.exists(npy_file_path):
        user_input = input("The file already exists. Do you want to continue and overwrite it? (y/n): ")
        if user_input.lower() == 'n':
            return None
        if user_input.lower() != 'y' and user_input.lower() != 'n':
            print('Not the correct response. Please type "y" or "n"')

    np.save(npy_file_path, data)
    print(f"Data saved to {npy_file_path}")


def read_from_npy(npy_file_path):
    """
    Read data from a NumPy file (.npy).

    Args:
        npy_file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray or None: The loaded data if the file exists, None otherwise.
    """
    if not os.path.exists(npy_file_path):
        print(f"The file {npy_file_path} does not exist.")
        return None
    data = np.load(npy_file_path, allow_pickle=True)
    return data


def merge_save_npz(out_path, new_arrays, drop_keys=()):
    """Merge ``new_arrays`` into an existing ``.npz`` and rewrite it.

    Loads any arrays already stored at ``out_path`` (so entries written by
    earlier passes are kept), overwrites/adds the keys in ``new_arrays``, and
    removes any keys listed in ``drop_keys``.
    Lets several results accumulate in one file instead of one file per pass.

    Args:
        out_path (str): The ``.npz`` file to update (created if missing).
        new_arrays (dict): Mapping of key -> array to add or overwrite.
        drop_keys (iterable of str, optional): Keys to delete from the merged
            archive if present.
    """
    arrays = {}
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            arrays = {k: d[k] for k in d.files}
    arrays.update(new_arrays)
    for key in drop_keys:
        arrays.pop(key, None)
    np.savez(out_path, **arrays)


def find_run_hdf5(data_dir, run_num):
    """Locate the single HDF5 file matching ``<run_num>-*.hdf5`` in ``data_dir``.

    Raises :class:`FileNotFoundError` if no matching file exists and
    :class:`ValueError` if multiple files match ``run_num``.
    """
    hits = sorted(glob.glob(os.path.join(data_dir, f"{run_num}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {run_num} in {data_dir}")
    if len(hits) > 1:
        raise ValueError(f"multiple HDF5 files for run {run_num}: {hits}")
    return hits[0]


def pack_npz_entries(prefix, mapping):
    """Pack a dict of Python/NumPy scalars and arrays under ``{prefix}__{k}``.

    Entries with value ``None`` are skipped. Preserves existing ``ndarray``
    dtypes and normalizes Python ``str``/``bool``/``int``/``float`` scalars to
    explicit NumPy scalar types so ``np.savez(..., allow_pickle=False)`` works
    cleanly.
    """
    packed = {}
    pfx = f"{prefix}__" if prefix else ""
    for k, v in mapping.items():
        if v is None:
            continue
        key = f"{pfx}{k}"
        if isinstance(v, np.ndarray):
            packed[key] = v
        elif isinstance(v, str):
            packed[key] = np.str_(v)
        elif isinstance(v, (bool, np.bool_)):
            packed[key] = np.bool_(v)
        elif isinstance(v, (int, np.integer)):
            packed[key] = np.int64(v)
        elif isinstance(v, (float, np.floating)):
            packed[key] = np.float64(v)
        else:
            packed[key] = np.asarray(v)
    return packed


def pack_nested_npz(prefix, nested):
    """Pack a two-level dict ``{sub_id: {k: v}}`` under ``{prefix}__{sub_id}__{k}``."""
    packed = {}
    for sub_id, fields in nested.items():
        sub_prefix = f"{prefix}__{sub_id}" if prefix else str(sub_id)
        packed.update(pack_npz_entries(sub_prefix, fields))
    return packed


def npz_settings_mismatch(
    npz_data,
    prefix,
    want,
    *,
    allow_missing_keys=False,
    fallback_prefix=None,
    fallback_aliases=None,
):
    """Compare stored settings in ``npz_data`` against ``want`` -> list of diff strings.

    For each ``(field, val)`` in ``want``, looks up ``f"{prefix}__{field}"``
    (or bare ``field`` when ``prefix`` is empty/None). When a key is absent and
    ``fallback_prefix`` is given, checks ``f"{fallback_prefix}__{alias}"``
    (mapping ``field`` through ``fallback_aliases`` when provided).

    ``val is None`` checks only key presence (skipped if ``allow_missing_keys``).
    Supports ``np.ndarray``, ``bool``, ``int``, ``str``, and ``float`` values.
    """
    diffs = []
    aliases = fallback_aliases or {}
    for field, val in want.items():
        k_primary = f"{prefix}__{field}" if prefix else str(field)
        if k_primary in npz_data.files:
            k_use = k_primary
        elif fallback_prefix is not None:
            fb_field = aliases.get(field, field)
            k_fb = f"{fallback_prefix}__{fb_field}" if fallback_prefix else str(fb_field)
            k_use = k_fb if k_fb in npz_data.files else None
        else:
            k_use = None

        if k_use is None:
            if not allow_missing_keys:
                req = f" (requested {val:g})" if isinstance(val, (int, float)) and not isinstance(val, bool) else ""
                diffs.append(f"{field}: missing in npz{req}")
            continue
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            got = np.asarray(npz_data[k_use])
            if got.shape != val.shape or not np.allclose(got, val, rtol=1e-6, atol=1e-12, equal_nan=True):
                diffs.append(f"{field}: stored array differs from requested")
        elif isinstance(val, (bool, np.bool_)):
            got = bool(npz_data[k_use])
            if got != bool(val):
                diffs.append(f"{field}: stored {got} != {bool(val)}")
        elif isinstance(val, (int, np.integer)):
            got = int(npz_data[k_use])
            if got != int(val):
                diffs.append(f"{field}: stored {got} != {int(val)}")
        elif isinstance(val, str):
            got = str(npz_data[k_use])
            if got != val:
                diffs.append(f"{field}: stored {got!r} != {val!r}")
        else:
            got = float(npz_data[k_use])
            if not np.isclose(got, float(val), rtol=1e-9, atol=1e-15):
                diffs.append(f"{field}: stored {got:g} != {float(val):g}")
    return diffs


def prepare_incremental_npz(
    out_path,
    run_ids,
    marker_suffix,
    *,
    overwrite=False,
    full_overwrite=False,
    prefix_fmt="{r}__",
    want=None,
    want_for_run=None,
    mismatch_kwargs=None,
    label="run",
):
    """Inspect an existing ``.npz`` cache for incremental per-run batch updates.

    - If ``overwrite and full_overwrite``: deletes ``out_path`` and returns
      ``(set(), [])``.
    - If ``overwrite`` (partial): collects ``drop_keys`` matching ``prefix_fmt``
      for ``run_ids`` and removes ``run_ids`` from ``existing_runs`` so only
      those runs are recomputed while unrequested runs stay intact.
    - If not ``overwrite``: validates each requested run already in
      ``existing_runs`` against ``want_for_run(npz_data, r)`` (or ``want``) via
      :func:`npz_settings_mismatch`, raising :class:`ValueError` on any diff.

    Returns:
        tuple[set[str], list[str]]: ``(existing_runs, drop_keys)``.
    """
    if overwrite and full_overwrite and os.path.isfile(out_path):
        os.remove(out_path)
        return set(), []

    existing_runs = set()
    drop_keys = []
    if not os.path.isfile(out_path):
        return existing_runs, drop_keys

    mkw = dict(mismatch_kwargs or {})
    with np.load(out_path) as d:
        existing_runs = {
            k.split("__", 1)[0] for k in d.files if k.endswith(marker_suffix)
        }
        if overwrite:
            prefixes = tuple(prefix_fmt.format(r=r) for r in run_ids)
            drop_keys = [k for k in d.files if k.startswith(prefixes)]
            existing_runs -= set(run_ids)
        elif want is not None or want_for_run is not None:
            for r in run_ids:
                if r not in existing_runs:
                    continue
                run_want = want_for_run(d, r) if want_for_run is not None else want
                if not run_want:
                    continue
                pfx = prefix_fmt.format(r=r).rstrip("_")
                r_mkw = dict(mkw)
                if "fallback_prefix_fmt" in r_mkw:
                    r_mkw["fallback_prefix"] = r_mkw.pop("fallback_prefix_fmt").format(r=r).rstrip("_")
                diffs = npz_settings_mismatch(d, pfx, run_want, **r_mkw)
                if diffs:
                    raise ValueError(
                        f"{out_path}: stored {label} '{r}' differs "
                        f"from requested settings ({'; '.join(diffs)}); "
                        "pass overwrite=True to regenerate"
                    )
    return existing_runs, drop_keys


_WIDTH_RE = re.compile(r"(\d+)")


def fmt_num(val, spec, missing="---", *, scale=1.0):
    """Format a finite numeric scalar with ``spec``, or right-aligned ``missing``."""
    v = float(val) * float(scale)
    if np.isfinite(v):
        return format(v, spec)
    m = _WIDTH_RE.search(spec)
    width = int(m.group(1)) if m else len(missing)
    return f"{missing:>{width}}"
