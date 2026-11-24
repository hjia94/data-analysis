"""Jun-2026 runs 08-14: calibrated Mach-probe flow along a P33 line scan.

The line-scan sibling of :mod:`Jun2026_mach_flow` (run 32, a plane). Same
reduction -- read six tips, apply the ``17-20`` area ratios, get M and (with an
assumed T_e) v -- but the moving probe traces a 1-D line, not a plane, so there
is no rotation centre to fit and no azimuthal component: the product is
``v_X``/``v_Y``/``v_Z`` per position, per time bin.

What is different from run 32, and why:

* **Bins are the paired IV run's sweeps, not a uniform grid.** Runs 08-14 carry
  no swept tip, so T_e is not measured here either -- but the plan is to replace
  the assumed T_e with the *measured* T_e from the puff-matched Langmuir run
  (:data:`MACH_TO_IV`). Averaging Mach between that run's sweep timestamps makes
  the two line up one bin per sweep, so the measured-T_e swap is later a
  per-bin multiply rather than a re-binning. See :func:`sweep_bin_edges`.
* **1-D positions.** The scan is a line, so the reduction grids nothing; it
  keeps the scanned axis as ``pos`` and records which axis moved.

Velocities assume a constant T_e (:data:`TE_EV`); ``v = M * c_s`` scales as
``sqrt(T_e)``. That is a stated scale, not a measurement -- recorded in the npz
and printed on every figure. Mach number is the measured quantity and is stored
beside it, so a later measured-T_e pass rescales v without re-reading the run.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_mach_line.py
"""

import datetime
import os

import re

import numpy as np
from tqdm import tqdm

import Jun2026_IV as jiv
import Jun2026_mach_flow as jmf
from data_analysis.io import open_lapd, parse_shunts, position_shots
from data_analysis.io.probe_map import channel_wiring
from data_analysis.plasma.mach import (K_HUTCHINSON, binned_face_ratio,
                                       flow_velocity, mach_single)
from data_analysis.utils import run_num_of

DATA_DIR = jmf.DATA_DIR
CAL_NPZ = jmf.CAL_NPZ

#: Mach line run -> the IV run whose sweep timing bins it, matched by gas puff
#: (both 23..45 ms, valve 75). Read from each run's own description; the Mach
#: runs ascend in puff (08 = 23 ms) while the IV runs do not (00 = 24 ms, 06 =
#: 23 ms), so this is a puff match, not a run-number offset. Zero-padded strings
#: -- the run number the whole pipeline addresses npz by (:func:`run_num_of`),
#: not an int, which builds the wrong ``<run>-sweep-data.npz`` stem.
MACH_TO_IV = {"08": "06", "09": "00", "10": "01", "11": "02",
              "12": "03", "13": "04", "14": "05"}

#: Preferred axis order when a run wires several. A run carries only the axes it
#: wired -- the line runs 08-14 have X, Y (no axial Z tips; those were added for
#: the run-32 plane), a later run may have all three -- so the axes are
#: discovered per run (:func:`discover_axes`) and ordered by this, never fixed.
AXIS_ORDER = ("X", "Y", "Z")

# Tip name at the end of a wiring description: 'Isat, Mach@P33-X+' -> 'X+'.
# Same rule as Jun2026_mach_flow/cal: splitting on '-' fails, the sign is a '-'.
_TIP_RE = re.compile(r"([XYZ][+-])\s*$", re.IGNORECASE)

#: Assumed electron temperature [eV] and ion mass [proton masses; He = 4]. NOT
#: measured in these runs -- see the module docstring.
TE_EV = jmf.TE_EV
ION_MU = jmf.ION_MU

NPZ_SUFFIX = "-mach-flow-line-data.npz"


def line_npz_path(ifn):
    """Co-located npz beside the raw HDF5, named from the run number."""
    return os.path.join(os.path.dirname(ifn), f"{run_num_of(ifn)}{NPZ_SUFFIX}")


def tip_channels(ifn):
    """``{'X+': ('machscope', 'C1'), ...}`` for every Mach tip the run wires.

    Carries the **scope** with each channel, not just the channel: the scope
    group's name drifts across the campaign (``machscope`` on the 00-14 line
    runs, a single ``scope`` on 16+), so it is resolved from the wiring per run
    (:func:`mach_scope`), never fixed. Unlike :func:`Jun2026_mach_flow.tip_channels`
    this also demands no fixed set of six -- a line run wires only the axes it
    measured -- so completeness is judged per axis by :func:`discover_axes`.

    Never a hardcoded C1=X+ table: a swapped +/- assignment inverts R and flips
    the sign of every flow vector, which looks entirely plausible.
    """
    out = {}
    for (scope, chan), desc in channel_wiring(ifn).items():
        m = _TIP_RE.search(desc)
        if m:
            out[m.group(1).upper()] = (scope, chan)
    return out


def mach_scope(chans):
    """The one scope group the Mach tips live on -> ``str``.

    Raises if the wired tips span more than one scope: this module reads every
    tip from one ``scope_name``, so a split set would silently read only some.
    """
    scopes = {scope for scope, _ in chans.values()}
    if len(scopes) != 1:
        raise ValueError(
            f"Mach tips span scopes {sorted(scopes)}; expected one. This module "
            "reads all tips from a single scope group.")
    return scopes.pop()


def discover_axes(chans):
    """Axes with *both* faces wired, in :data:`AXIS_ORDER` -> ``list[str]``.

    An axis missing one face cannot form a ratio, so it is not a measured axis;
    it is dropped here rather than half-read. Raises if no axis is complete,
    since then there is no flow to compute.
    """
    axes = [a for a in AXIS_ORDER if f"{a}+" in chans and f"{a}-" in chans]
    if not axes:
        raise ValueError(
            f"no axis has both faces wired (tips present: {sorted(chans)}); "
            "nothing to form a Mach ratio from.")
    return axes


def iv_sweep_timestamps(data_dir, iv_run):
    """The paired IV run's per-sweep midpoint times [s] -> ``(n_sweeps,)``.

    ``data_timestamp`` is the sweep midpoint (:func:`langmuir.prepare_sweep_data`),
    stored per tip; every tip of a run shares the sweep grid, so any tip's is the
    run's. Returned on the scope time base the sweeps were found on -- the Mach
    run's own time axis is shifted onto the same base by ``searchsorted``, never
    the other way.
    """
    from data_analysis.plasma.langmuir import sweep_npz_paths
    sweep_path, _ = sweep_npz_paths(data_dir, iv_run)
    with np.load(sweep_path) as d:
        # Any tip's grid is the run's; take the first data_timestamp key rather
        # than resolving a tip name, since none of them is preferred here.
        key = next(k for k in d.files if k.endswith("/data_timestamp"))
        return np.asarray(d[key], float)


def sweep_bin_edges(tarr, timestamps):
    """Sample-index edges binning ``tarr`` between consecutive sweep midpoints.

    ``timestamps`` are the IV sweep midpoints [s]; bin ``j`` spans
    ``[timestamps[j], timestamps[j+1])`` and is **labelled by its upper edge**,
    sweep ``j+1`` -- so ``n_sweeps`` sweeps give ``n_sweeps-1`` bins, the first
    dropped and the last kept. Contiguous by construction (each bin ends where
    the next begins), so :func:`binned_face_ratio` consumes them unchanged.

    Returns ``(edges, upper_idx)``: the ``searchsorted`` indices into ``tarr``
    (clamped to ``[0, nt]`` so an out-of-record sweep bounds an empty bin rather
    than indexing past the record), and the sweep index (``j+1``) each bin
    carries -- the key a measured-T_e pass joins on.

    Simple contiguous binning: a bin whose timestamps fall outside the Mach
    record's -2..8 ms collapses to zero width and yields NaN, left in place
    rather than dropped. Edge handling (partial coverage, empty bins) is
    deliberately not refined here.
    """
    tarr = np.asarray(tarr, float)
    edges = np.clip(np.searchsorted(tarr, np.asarray(timestamps, float)),
                    0, tarr.size)
    upper_idx = np.arange(1, timestamps.size)
    return edges, upper_idx


def scanned_axis(pos):
    """The line's moving axis -> ``(pos_1d, axis_name)``.

    A line scan holds one of x/y fixed; the moving one is the position axis.
    Raises if both move (a plane) or neither does, since either breaks the 1-D
    product this module writes.
    """
    moving = [(name, ax) for name, ax in (("x", pos.xpos), ("y", pos.ypos))
              if ax.size > 1]
    if len(moving) != 1:
        got = {"x": pos.xpos.size, "y": pos.ypos.size}
        raise ValueError(
            f"expected exactly one scanned axis for a line, got {got}; "
            "this is a plane or a single point, not a line scan.")
    name, ax = moving[0]
    return np.asarray(ax, float), name


def batch_line(ifn, iv_run, cal_path=CAL_NPZ, te_ev=TE_EV, out_path=None):
    """Read one Mach line, bin by the paired IV sweeps, apply kappa -> npz path.

    ``iv_run`` is the puff-matched Langmuir run supplying the sweep timestamps
    (:data:`MACH_TO_IV`). Writes ``<run>-mach-flow-line-data.npz``: ``M_*`` and
    ``v_*`` [km/s] as ``(npos, nbin)``, the amplitude and valid-sample count per
    bin (:func:`binned_face_ratio` -- the noise-floor evidence, ungated here),
    the 1-D position axis, each bin's ``iv_sweep_index``, and the calibration and
    T_e applied.
    """
    kappa, kappa_err, method = jmf.load_calibration(cal_path)
    chans = tip_channels(ifn)
    axes = discover_axes(chans)
    scope = mach_scope(chans)
    # kappa comes from the run-32 six-tip calibration; a line run measuring a
    # subset must still find its axes there (the calibration binds to the probe,
    # not the run), so check before reading 10 GB.
    missing_cal = [a for a in axes if a not in kappa]
    if missing_cal:
        raise ValueError(f"{cal_path}: calibration has no kappa for {missing_cal}")

    run = open_lapd(ifn)
    shunts = parse_shunts(run.description().raw)
    # Only the wired tips need a shunt; an unused face's absence is not an error.
    tips = [f"{a}{s}" for a in axes for s in "+-"]
    no_shunt = [t for t in tips if t not in shunts]
    if no_shunt:
        # Never default: the tips differ (300 ohm on X-, 75/43 elsewhere), so a
        # guessed value yields a wrong current with nothing complaining.
        raise ValueError(f"{ifn}: no shunt in description for {no_shunt}")

    pos = jiv.read_lp_positions(ifn)
    pos_1d, axis_name = scanned_axis(pos)
    tarr = run.time_array(scope_name=scope)

    timestamps = iv_sweep_timestamps(os.path.dirname(ifn), iv_run)
    edges, upper_idx = sweep_bin_edges(tarr, timestamps)
    # Trim trailing edges whose bin starts past the end of the Mach record so
    # trailing all-empty bins are dropped rather than padded onto the output.
    last = int(np.searchsorted(edges, tarr.size, side="left"))
    edges = edges[:max(last + 1, 2)]
    nbin = edges.size - 1
    upper_idx = upper_idx[:nbin]
    # Bin midpoint [ms] labels the frame; the sweep index it carries (upper edge)
    # is what a measured-T_e pass joins on.
    t_ms = 0.5 * (timestamps[:nbin] + timestamps[1:nbin + 1]) * 1e3

    mach = {a: np.full((pos.npos, nbin), np.nan) for a in axes}
    amp = np.full((len(axes), pos.npos, nbin), np.nan)
    n_valid = np.zeros((len(axes), pos.npos, nbin), dtype=np.int32)

    skipped = []
    for p in tqdm(range(pos.npos), desc=f"Mach line {run_num_of(ifn)}", unit="pos"):
        shots = position_shots(pos.shot_nums, p, pos.nshot)
        for i, axis in enumerate(axes):
            stacks = []
            for sign in "+-":
                _scope, chan = chans[f"{axis}{sign}"]
                stack, _ = run.channel(chan, scope_name=_scope, shots=shots)
                # A position the scope never wrote comes back as None: leave its
                # row NaN and record it, so an unwritten position is told apart
                # from one that measured zero flow.
                if stack is None:
                    skipped.append((p, f"{axis}{sign}"))
                    stacks = None
                    break
                stack /= shunts[f"{axis}{sign}"]
                stacks.append(stack)
            if stacks is None:
                continue
            R, counts, amplitude = binned_face_ratio(stacks[0], stacks[1], edges)
            mach[axis][p] = mach_single(R, kappa[axis])
            amp[i, p] = amplitude
            n_valid[i, p] = counts

    lost = sorted({p for p, _ in skipped})
    if lost:
        print(f"\n  {len(lost)} position(s) had no scope data and stay NaN: "
              f"{lost[:10]}{' ...' if len(lost) > 10 else ''}")

    arrays = {
        "pos": pos_1d, "pos_axis": np.str_(axis_name), "t_ms": t_ms,
        # The sweep each bin was averaged over (upper edge), so a measured-T_e
        # pass joins Mach to Te by index rather than re-matching floats.
        "iv_sweep_index": upper_idx.astype(np.int32),
        "iv_run": np.str_(iv_run),
        "n_missing_positions": np.int32(len(lost)),
        "n_valid": n_valid, "amplitude": amp,
        # 'axes' is provenance AND the record of which axes this run measured
        # (a subset for a 4-tip line run): a reader learns the M_*/v_* key order
        # and the panel set from it, without assuming three.
        "axes": np.array(axes),
        "kappa": np.array([kappa[a] for a in axes]),
        "kappa_err_sys": np.array([kappa_err[a] for a in axes]),
        "kappa_method": np.array([method[a] for a in axes]),
        "shunts": np.array([shunts[f"{a}{s}"] for a in axes for s in "+-"]),
        "tip_channels": np.array([f"{a}{s}={chans[f'{a}{s}'][0]}/{chans[f'{a}{s}'][1]}"
                                  for a in axes for s in "+-"]),
        "te_ev": np.float64(te_ev), "ion_mu": np.float64(ION_MU),
        "K_hutchinson": np.float64(K_HUTCHINSON),
        "nshot": np.int32(pos.nshot),
        "source_file": np.str_(os.path.basename(ifn)),
        "calibration_file": np.str_(os.path.basename(cal_path)),
        "created": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    for a in axes:
        arrays[f"M_{a}"] = mach[a]
        arrays[f"v_{a}"] = flow_velocity(mach[a], te_ev, ION_MU)

    out_path = out_path or line_npz_path(ifn)
    np.savez(out_path, **arrays)
    print(f"\nWrote {out_path} ({pos.npos} positions x {nbin} sweep bins from "
          f"IV run {iv_run}, T_e = {te_ev} eV assumed)")
    return out_path


def line_ifn(data_dir, mach_run):
    """The raw HDF5 for a Mach line run (zero-padded string), by its prefix."""
    import glob
    hits = glob.glob(os.path.join(data_dir, f"{mach_run}-*Mach*line*.hdf5"))
    if not hits:
        raise FileNotFoundError(f"no Mach line HDF5 for run {mach_run} in {data_dir}")
    return hits[0]


def batch_all(data_dir=DATA_DIR, mapping=None, overwrite=False):
    """Batch every mapped Mach line run -> written npz paths.

    Existing npz files are skipped unless ``overwrite=True`` (use ``overwrite``
    to regenerate after calibration or binning changes).
    """
    mapping = mapping or MACH_TO_IV
    out = []
    for mach_run, iv_run in sorted(mapping.items()):
        ifn = line_ifn(data_dir, mach_run)
        npz = line_npz_path(ifn)
        if not overwrite and os.path.exists(npz):
            print(f"  run {mach_run}: {os.path.basename(npz)} exists, skipping")
            out.append(npz)
            continue
        out.append(batch_line(ifn, iv_run))
    return out


if __name__ == "__main__":
    batch_all()
