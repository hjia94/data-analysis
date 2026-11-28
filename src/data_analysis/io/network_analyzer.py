"""Read Network Analyzer (VNA) exports (Keysight/Agilent N5230C and HP/Agilent E5100A).

Moved here from ``read/read_network_analyzer_data.py`` during the package reorg;
the readers for VNA CSV/TXT/DAT exports belong with the other format parsers in
``data_analysis.io``.

Parsers return raw VNA data verbatim (no bench amplifier or attenuator math is
applied here). For B-dot Helmholtz-bench calibration physics, divide the raw
transfer ratio by any bench gain/attenuator to obtain real probe-terminal signal
level and pass it to :mod:`data_analysis.plasma.bdot`.

Verified against real Agilent N5230C exports under
``D:/data/LAPD/Bdot_calibration/...`` and real Agilent E5100A ``Bdot-10T-C16``
calibration files on pandas 3.0 / numpy 2.x.
"""

from __future__ import annotations

import os
import re
import struct
import warnings

import numpy as np
import pandas as pd

_AXES = ("X", "Y", "Z")
_E5100A_BIN_HEADER_SIZE = 0x126


def read_NA_data(filepath, verbose=False):
    """Read Keysight/Agilent N5230C S-parameter data from a CSV export.

    Args:
        filepath (str): Path to the CSV file.
        verbose (bool): If True, print detailed debug information.

    Returns:
        dict: Dictionary containing ``metadata``, ``Freq(Hz)``, and complex
            S-parameters keyed by base name (e.g. ``'S21'``).
    """
    metadata = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as file:
        for _ in range(5):
            metadata.append(file.readline().strip())

    # Row 8 (0-based index 7) is the column header.
    data = pd.read_csv(filepath, skiprows=7, header=0)
    column_names = data.columns.tolist()

    if verbose:
        print("\nAll column names:")
        for i, col in enumerate(column_names):
            print(f"{i}: {repr(col)}")

    spectral_dict = {
        "metadata": {
            "ICSV_version": metadata[0],
            "instrument": metadata[1],
            "model": metadata[2],
            "datetime": metadata[3],
            "source": metadata[4],
        }
    }

    freq_data = data["Freq(Hz)"]
    end_idx = freq_data[freq_data == "END"].index
    if len(end_idx) == 0:
        raise ValueError("No 'END' marker found in frequency data")
    end_idx = end_idx[0]

    freq_data = freq_data[:end_idx]
    spectral_dict["Freq(Hz)"] = freq_data.astype(float).values

    for col in column_names:
        if "REAL" in col and col.startswith("S"):
            if verbose:
                print(f"\nProcessing column: {repr(col)}")
            base_name = col.split("REAL")[0].strip(" ()")
            if verbose:
                print(f"Base name: {repr(base_name)}")

            imag_col = None
            for potential_imag in column_names:
                if potential_imag.startswith(base_name) and "IMAG" in potential_imag:
                    imag_col = potential_imag
                    break

            if imag_col:
                if verbose:
                    print("Found matching pair:")
                    print(f"  Real: {repr(col)}")
                    print(f"  Imag: {repr(imag_col)}")
                real_data = data[col].values[:end_idx]
                imag_data = data[imag_col].values[:end_idx]
                spectral_dict[base_name] = real_data + 1j * imag_data
                if verbose:
                    print(f"Added {base_name} to dictionary")
            elif verbose:
                print(f"No matching imaginary column found for {repr(col)}")

    if verbose:
        print("\nDictionary keys:", spectral_dict.keys())

    return spectral_dict


def _arrays_from_e5100a_format(main_arr, sub_arr, format_type, source=""):
    """Convert E5100A (Main, Sub) arrays into ``(mag, phase_deg, complex_response)``."""
    fmt = (format_type or "LINMP").strip().upper()
    if fmt == "LINMP":
        mag = main_arr
        phase_deg = sub_arr
        complex_response = mag * np.exp(1j * np.deg2rad(phase_deg))
    elif fmt == "LOGMP":
        mag = 10.0 ** (main_arr / 20.0)
        phase_deg = sub_arr
        complex_response = mag * np.exp(1j * np.deg2rad(phase_deg))
    elif fmt == "REIM":
        complex_response = main_arr + 1j * sub_arr
        mag = np.abs(complex_response)
        phase_deg = np.rad2deg(np.angle(complex_response))
    else:
        where = f" in {source}" if source else ""
        raise ValueError(f"Unsupported E5100A FORMAT TYPE {fmt!r}{where}")
    return mag, phase_deg, complex_response


def read_e5100a_csv(filepath):
    """Read an Agilent/HP E5100A ASCII export (``.csv`` or ``.txt``).

    Handles both tab-delimited (standard E5100A ``.CSV``/``.txt`` exports) and
    comma-delimited files, parses quoted header metadata case-insensitively
    (e.g. ``NUMBER of POINTS``), and converts ``LINMP``, ``LOGMP``, or ``REIM``
    columns into complex ``V_A / V_R`` as recorded by the VNA (no bench gain
    removed).

    Note: The E5100A internal clock is often unset (e.g. ``DATE: Apr 7 2005``
    on 2026 files); do not rely on ``metadata['DATE']`` for chronology.

    Returns:
        dict with keys ``freq_hz``, ``main``, ``sub``, ``mag``, ``phase_deg``,
        ``complex_response``, and ``metadata``.
    """
    metadata = {}
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('"'):
                unquoted = line.strip('"')
                if ":" in unquoted:
                    key, val = unquoted.split(":", 1)
                    metadata[key.strip().upper()] = val.strip()
                elif "INSTRUMENT" not in metadata and "E5100" in unquoted.upper():
                    metadata["INSTRUMENT"] = unquoted.strip()
                continue
            parts = re.split(r"[\t,]+", line)
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric E5100A data rows found in {filepath}")

    data = np.asarray(rows, dtype=float)
    freq_hz = data[:, 0]
    main_arr = data[:, 1]
    sub_arr = data[:, 2]

    meas_type = metadata.get("MEASURE TYPE", "").upper()
    if meas_type and meas_type != "AR":
        warnings.warn(
            f"{os.path.basename(filepath)}: MEASURE TYPE is {meas_type!r} (expected 'AR')",
            UserWarning,
            stacklevel=2,
        )

    mag, phase_deg, complex_response = _arrays_from_e5100a_format(
        main_arr, sub_arr, metadata.get("FORMAT TYPE", "LINMP"), source=str(filepath)
    )
    return {
        "freq_hz": freq_hz,
        "main": main_arr,
        "sub": sub_arr,
        "mag": mag,
        "phase_deg": phase_deg,
        "complex_response": complex_response,
        "metadata": metadata,
    }


def _unpack_e5100a_record(fdata, pos):
    """Parse one binary E5100A data record -> ``(array, next_pos)``."""
    if pos + 10 > len(fdata):
        raise ValueError("Truncated E5100A binary record header")
    _x = struct.unpack_from("=L", fdata, pos)[0]
    pos += 4
    npts = struct.unpack_from("=H", fdata, pos)[0]
    pos += 2
    _y = struct.unpack_from("=L", fdata, pos)[0]
    pos += 4
    nbytes = npts * 8
    if pos + nbytes > len(fdata):
        raise ValueError(f"Truncated E5100A binary record: expected {nbytes} bytes for {npts} points")
    arr = np.frombuffer(fdata, dtype="<f8", count=npts, offset=pos).copy()
    return arr, pos + nbytes


def read_e5100a_binary(filepath, *, freq_hz=None, use_alt_channels=False, format_type="REIM"):
    """Read an Agilent/HP E5100A binary ``.DAT`` export (DATA ONLY format).

    DATA ONLY binary exports do not store the frequency sweep; pass ``freq_hz``
    when known. Ported from ``E5100A_Reader (1).py``.
    """
    with open(filepath, "rb") as f:
        fdata = f.read()

    if len(fdata) < _E5100A_BIN_HEADER_SIZE:
        raise ValueError(f"{filepath}: file shorter than E5100A header ({_E5100A_BIN_HEADER_SIZE} bytes)")

    f_lo, f_hi = struct.unpack_from("=dd", fdata, 0)
    if f_lo != 10000.0 or f_hi != 180000000.0:
        warnings.warn(
            f"{os.path.basename(filepath)}: unexpected E5100A binary header bounds ({f_lo:g}, {f_hi:g})",
            UserWarning,
            stacklevel=2,
        )

    pos = _E5100A_BIN_HEADER_SIZE
    main_arr, pos = _unpack_e5100a_record(fdata, pos)
    alt_ch_0, pos = _unpack_e5100a_record(fdata, pos)
    sub_arr, pos = _unpack_e5100a_record(fdata, pos)
    alt_ch_1, _ = _unpack_e5100a_record(fdata, pos)

    if use_alt_channels:
        main_arr, sub_arr = alt_ch_0, alt_ch_1

    if main_arr.size != sub_arr.size:
        raise ValueError(
            f"{filepath}: E5100A MAIN/SUB array lengths differ ({main_arr.size} != {sub_arr.size})"
        )

    freq_out = None
    if freq_hz is not None:
        freq_out = np.asarray(freq_hz, dtype=float)
        if freq_out.ndim != 1 or freq_out.size != main_arr.size:
            raise ValueError(
                f"freq_hz must be 1-D with {main_arr.size} points, got shape {freq_out.shape}"
            )

    mag, phase_deg, complex_response = _arrays_from_e5100a_format(
        main_arr, sub_arr, format_type, source=str(filepath)
    )
    return {
        "freq_hz": freq_out,
        "main": main_arr,
        "sub": sub_arr,
        "mag": mag,
        "phase_deg": phase_deg,
        "complex_response": complex_response,
        "metadata": {
            "HEADER_FMIN_HZ": f_lo,
            "HEADER_FMAX_HZ": f_hi,
            "FORMAT TYPE": format_type.upper(),
        },
    }


def load_bdot_vna_dir(directory, probe_id=None, *, ext=(".csv", ".txt")):
    """Discover and load 3-file (diagonal) or 9-file (3x3) E5100A B-dot calibration exports.

    Matches case-insensitive ``<probe_id>B<drive>P<probe>`` files (tolerating
    optional trailing tags such as ``C16BXPX-12-24.csv`` or ``C16BXPX(1).csv``).
    Maps file ``B<drive>P<probe>`` to matrix entry ``(probe_idx, drive_idx)``
    matching ``V_coil = A @ dB/dt`` (rows = probe axes ``Px, Py, Pz``,
    cols = driven field axes ``Bx, By, Bz``).

    Returns:
        dict with keys:
        - ``probe_id``: str (passed ``probe_id`` or inferred prefix)
        - ``axes``: ``('X', 'Y', 'Z')``
        - ``freq_hz``: 1-D shared frequency axis [Hz]
        - ``H_vna``: complex ndarray of shape ``(3, 3, nf)``, raw VNA ``V_A / V_R``
          with ``nan + 0j`` for any absent off-diagonal entries
        - ``files``: ``{(probe_idx, drive_idx): filepath}``
        - ``records``: ``{(probe_idx, drive_idx): read_e5100a_csv(filepath)}``
        - ``is_diagonal_only``: bool (True when only diagonal files are present)
    """
    directory = os.fspath(directory)
    exts_clean = tuple(e.lstrip(".").lower() for e in ext)
    ext_alt = "|".join(re.escape(e) for e in exts_clean)
    if probe_id:
        pat = re.compile(
            rf"^(?P<prefix>{re.escape(probe_id)})[_-]?B(?P<drive>[XYZ])P(?P<probe>[XYZ])(?![A-Za-z]).*\.(?:{ext_alt})$",
            re.IGNORECASE,
        )
    else:
        pat = re.compile(
            rf"^(?P<prefix>.*?)B(?P<drive>[XYZ])P(?P<probe>[XYZ])(?![A-Za-z]).*\.(?:{ext_alt})$",
            re.IGNORECASE,
        )

    axis_idx = {a: i for i, a in enumerate(_AXES)}
    found = {}
    prefixes = set()
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if not os.path.isfile(full):
            continue
        m = pat.match(entry)
        if not m:
            continue
        drive = m.group("drive").upper()
        probe = m.group("probe").upper()
        key = (axis_idx[probe], axis_idx[drive])
        if key in found:
            raise ValueError(
                f"Multiple files match B{drive}P{probe} in {directory}: "
                f"{os.path.basename(found[key])}, {entry}"
            )
        found[key] = full
        pfx = m.group("prefix").rstrip("_-")
        if pfx:
            prefixes.add(pfx)

    if not found:
        label = f" for probe {probe_id!r}" if probe_id else ""
        raise FileNotFoundError(
            f"No E5100A B-dot calibration files matching B[XYZ]P[XYZ]{label} found in {directory}"
        )

    missing_diag = [
        f"B{_AXES[i]}P{_AXES[i]}" for i in range(3) if (i, i) not in found
    ]
    if missing_diag:
        raise FileNotFoundError(
            f"{directory}: missing required diagonal calibration file(s): {missing_diag}"
        )

    records = {}
    freq_ref = None
    for key, path in sorted(found.items()):
        rec = read_e5100a_csv(path)
        if freq_ref is None:
            freq_ref = rec["freq_hz"]
        elif rec["freq_hz"].shape != freq_ref.shape or not np.allclose(
            rec["freq_hz"], freq_ref, rtol=1e-6, atol=1e-3
        ):
            raise ValueError(
                f"{os.path.basename(path)}: frequency grid differs from other files in {directory}"
            )
        records[key] = rec

    H_vna = np.full((3, 3, freq_ref.size), np.nan + 0j, dtype=complex)
    for (p_idx, d_idx), rec in records.items():
        H_vna[p_idx, d_idx] = rec["complex_response"]

    inferred_id = probe_id or (sorted(prefixes)[0] if len(prefixes) == 1 else "")
    is_diagonal_only = all(p == d for p, d in found)
    return {
        "probe_id": str(inferred_id),
        "axes": _AXES,
        "freq_hz": freq_ref,
        "H_vna": H_vna,
        "files": found,
        "records": records,
        "is_diagonal_only": bool(is_diagonal_only),
    }
