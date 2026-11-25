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
import warnings

import numpy as np
from tqdm import tqdm

import Jun2026_IV as jiv
import Jun2026_mach_flow as jmf
from data_analysis.io import open_lapd, parse_shunts, position_shots
from data_analysis.io.probe_map import channel_wiring
from data_analysis.plasma.flow import (antisymmetric_rotation_profile,
                                       radial_flow_shear)
from data_analysis.plasma.langmuir import load_plasma_data, load_sweep_axes
from data_analysis.plasma.mach import (FLOW_SIGN_PLUS_FACING_AXIS,
                                       K_HUTCHINSON, binned_face_ratio,
                                       flow_velocity, mach_single,
                                       rescale_mach_velocity)
from data_analysis.plasma.profiles import (local_drift_scales,
                                           radial_profile_gradients)
from data_analysis.utils import merge_save_npz, run_num_of

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
            # Confirmed P33 6-tip face normals: X+ -> +x (West), Y+ -> +y (Up),
            # Z+ -> South (verified by bias-off Z flow on Cal 17/19 & Run 32) ->
            # default FLOW_SIGN_PLUS_FACING_AXIS = -1 yields lab-frame velocities.
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
        "flow_sign": np.int64(FLOW_SIGN_PLUS_FACING_AXIS),
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


RESCALED_OUT_NAME = "mach_line_rescaled_jun2026.npz"
SHEAR_ANNULUS_R_CM = (3.0, 6.0)     # outer antisymmetric dipole band [cm]
INNER_WALL_R_CM = (1.5, 4.0)        # hollowing-boundary band [cm]
DEFAULT_X_CENTER_CM = 1.0
DEFAULT_B0_G = 800.0


def _load_paired_iv_fields(data_dir, iv_run, profiles_npz=None, tip="P29-R", *,
                           x_center_cm=DEFAULT_X_CENTER_CM, smooth_window=5,
                           smooth_poly=2, b0_g=DEFAULT_B0_G):
    """Load paired IV ``(x_cm, Te, Te_err, v_ExB_km_s, ...)`` with x_center guard."""
    prof_path = profiles_npz or os.path.join(data_dir, "profiles_summary_jun2026.npz")
    if os.path.isfile(prof_path):
        with np.load(prof_path) as d:
            if f"{iv_run}__Te" in d.files:
                xc_key = f"{iv_run}__x_center_cm"
                if xc_key in d.files:
                    stored_xc = float(d[xc_key])
                    if not np.isclose(stored_xc, x_center_cm, atol=1e-6, rtol=0.0):
                        raise ValueError(
                            f"{prof_path}: IV run '{iv_run}' was signed around "
                            f"x_center_cm={stored_xc:g}, which disagrees with requested "
                            f"x_center_cm={x_center_cm:g}; regenerate profiles or match x_center_cm"
                        )
                return {
                    "x_cm": np.asarray(d[f"{iv_run}__x_cm"], dtype=float),
                    "Te": np.asarray(d[f"{iv_run}__Te"], dtype=float),
                    "Te_err": np.asarray(d[f"{iv_run}__Te_err"], dtype=float),
                    "v_ExB_km_s": np.asarray(d[f"{iv_run}__v_ExB_m_s"], dtype=float) * 1e-3,
                    "t_mid_ms": np.asarray(d[f"{iv_run}__t_mid_ms"], dtype=float),
                    "ref_sweep_idx": int(d[f"{iv_run}__ref_sweep_idx"]),
                    "gas_puff_ms": float(d[f"{iv_run}__gas_puff_ms"]),
                    "profiles_file": os.path.basename(prof_path),
                }

    # Explicit fallback when profiles_summary_jun2026.npz is absent: compute
    # v_ExB directly from saved Vp so it is never silently all-NaN.
    warnings.warn(
        f"{prof_path} not found (or missing IV run {iv_run}); falling back to "
        f"{iv_run}-plasma-data.npz and computing E_r/B on the fly at B0={b0_g:g} G.",
        RuntimeWarning,
        stacklevel=2,
    )
    Vp, Te, ne, _, Te_err, _, t_ls = load_plasma_data(data_dir, iv_run, tip)
    xpos, _, _, _ = load_sweep_axes(data_dir, iv_run, tip)
    x_cm = np.asarray(xpos, dtype=float)
    grads = radial_profile_gradients(
        x_cm, ne, Te, Vp, x_center_cm=x_center_cm,
        smooth_window=smooth_window, polyorder=smooth_poly, axis=0,
    )
    scales = local_drift_scales(ne, Te, b0_g, er_v_per_cm=grads.er_v_per_cm)
    return {
        "x_cm": x_cm,
        "Te": np.asarray(Te, dtype=float),
        "Te_err": np.asarray(Te_err, dtype=float),
        "v_ExB_km_s": np.asarray(scales["v_ExB_m_s"], dtype=float) * 1e-3,
        "t_mid_ms": np.asarray(t_ls, dtype=float) * 1e3,
        "ref_sweep_idx": -1,
        "gas_puff_ms": np.nan,
        "profiles_file": f"{iv_run}-plasma-data.npz",
    }


def rescale_line_with_iv(mach_run, data_dir=DATA_DIR, *, profiles_npz=None,
                         onset_npz=None, x_center_cm=DEFAULT_X_CENTER_CM,
                         smooth_window=5, smooth_poly=2, te_mode="upper",
                         tip="P29-R"):
    """Join a Mach line onto its paired IV run via ``iv_sweep_index`` and rescale ``v``.

    Replaces the placeholder constant ``T_e`` in ``<mach_run>-mach-flow-line-data.npz``
    with the measured ``T_e(x, t)`` from ``MACH_TO_IV[mach_run]`` (using
    :func:`rescale_mach_velocity`, which follows ``mach.flow_velocity``'s
    cold-ion ``gamma = 5/3`` sound speed), propagates ``Te_err`` and calibration
    ``kappa_err_sys`` uncertainties, separates the antisymmetric rotational
    profile ``v_rot(r)`` from uniform background offsets, computes the
    cylindrical shearing rate ``gamma_E = r dOmega/dr``, and compares directly
    against the paired P29 ``E_r / B`` drift profile on **matched** radial bands.

    Also computes the Mach-space pre-bias increment ``dM_Y = M_Y - M_Y_pre``
    before multiplying by ``c_s(T_e)``: a constant area-ratio calibration error
    ``Delta kappa`` is additive in ``M`` (not ``v``), so subtracting in ``M``
    cancels ``Delta M * c_s(x)`` even when ``T_e(x)`` is asymmetric, whereas
    subtracting in ``v`` cancels a true pre-bias background velocity.

    ``te_mode`` controls which IV timestamp scales bin ``k`` (spanning sweeps
    ``swp - 1`` to ``swp``): ``"upper"`` uses ``Te[:, swp]`` (the bin's
    ``iv_sweep_index``); ``"midpoint"`` averages ``0.5 * (Te[:, swp-1] + Te[:, swp])``.
    """
    if te_mode not in ("upper", "midpoint"):
        raise ValueError(f"te_mode must be 'upper' or 'midpoint', got {te_mode!r}")

    ifn = line_ifn(data_dir, mach_run)
    npz_path = line_npz_path(ifn)
    with np.load(npz_path) as m:
        if "flow_sign" not in m.files or int(m["flow_sign"]) != int(FLOW_SIGN_PLUS_FACING_AXIS):
            got = int(m["flow_sign"]) if "flow_sign" in m.files else "missing"
            raise ValueError(
                f"{npz_path}: stored flow_sign={got!r} != {FLOW_SIGN_PLUS_FACING_AXIS} "
                "(unified lab-axis Mach convention); regenerate or migrate the line archive."
            )
        iv_run = str(m["iv_run"])
        x_mach = np.asarray(m["pos"], dtype=float)
        t_ms = np.asarray(m["t_ms"], dtype=float)
        swp_idx = np.asarray(m["iv_sweep_index"], dtype=int)
        axes = [str(a) for a in m["axes"]]
        ion_mu = float(m["ion_mu"])
        K_hutch = float(m["K_hutchinson"])
        kappa = np.asarray(m["kappa"], dtype=float)
        kappa_err = np.asarray(m["kappa_err_sys"], dtype=float)
        M_by_axis = {a: np.asarray(m[f"M_{a}"], dtype=float) for a in axes}

    iv = _load_paired_iv_fields(
        data_dir, iv_run, profiles_npz=profiles_npz, tip=tip,
        x_center_cm=x_center_cm, smooth_window=smooth_window, smooth_poly=smooth_poly,
    )
    x_iv = iv["x_cm"]

    def _align_x(arr_2d):
        if x_mach.shape == x_iv.shape and np.allclose(x_mach, x_iv, atol=1e-2):
            return arr_2d
        out = np.full((x_mach.size, arr_2d.shape[1]), np.nan, dtype=float)
        for j in range(arr_2d.shape[1]):
            fin = np.isfinite(arr_2d[:, j])
            if np.sum(fin) >= 2:
                out[:, j] = np.interp(x_mach, x_iv[fin], arr_2d[fin, j],
                                      left=np.nan, right=np.nan)
        return out

    Te_x = _align_x(iv["Te"])
    Te_err_x = _align_x(iv["Te_err"])
    vExB_x = _align_x(iv["v_ExB_km_s"])

    if te_mode == "upper":
        Te_bin = Te_x[:, swp_idx]
        Te_err_bin = Te_err_x[:, swp_idx]
        vExB_bin = vExB_x[:, swp_idx]
    else:
        lo_idx = np.maximum(swp_idx - 1, 0)
        Te_bin = 0.5 * (Te_x[:, lo_idx] + Te_x[:, swp_idx])
        Te_err_bin = 0.5 * np.sqrt(Te_err_x[:, lo_idx] ** 2 + Te_err_x[:, swp_idx] ** 2)
        vExB_bin = 0.5 * (vExB_x[:, lo_idx] + vExB_x[:, swp_idx])

    rescaled = {}
    for i, a in enumerate(axes):
        # Systematic Mach error from area-ratio calibration: dM = (K/2) * (dkappa/kappa).
        m_err_sys = (0.5 * K_hutch * (kappa_err[i] / kappa[i])
                     if kappa[i] > 0 else np.nan)
        v_a, v_a_err = rescale_mach_velocity(
            M_by_axis[a], Te_bin, ion_mu, te_err_ev=Te_err_bin, M_err=m_err_sys,
        )
        rescaled[f"v_{a}_te"] = v_a
        rescaled[f"v_{a}_te_err"] = v_a_err

    # Rotational (Y) profile decomposition & flow shear.
    MY = M_by_axis["Y"]
    vY = rescaled["v_Y_te"]
    pre_bins = (t_ms < -0.5) & np.any(np.isfinite(vY), axis=0)
    has_pre_bias = bool(np.any(pre_bins))
    if has_pre_bias:
        vY_pre = np.nanmedian(vY[:, pre_bins], axis=1, keepdims=True)
        MY_pre = np.nanmedian(MY[:, pre_bins], axis=1, keepdims=True)
        dvY = vY - vY_pre
        dvY_dM = rescale_mach_velocity(MY - MY_pre, Te_bin, ion_mu)
    else:
        warnings.warn(
            f"Mach run {mach_run}: no finite pre-bias bins (t_ms < -0.5 ms); "
            "leaving pre-bias baseline and flow increments (dv_Y_te, dv_Y_dM) as NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        vY_pre = np.full((x_mach.size, 1), np.nan, dtype=float)
        dvY = np.full_like(vY, np.nan, dtype=float)
        dvY_dM = np.full_like(vY, np.nan, dtype=float)

    r_cm, v_rot, v_sym = antisymmetric_rotation_profile(
        x_mach, vY, x_center_cm=x_center_cm, axis=0
    )
    _, dv_rot, dv_sym = antisymmetric_rotation_profile(
        x_mach, dvY, x_center_cm=x_center_cm, axis=0
    )
    _, dv_rot_dM, _ = antisymmetric_rotation_profile(
        x_mach, dvY_dM, x_center_cm=x_center_cm, axis=0
    )
    # Pair P29 E_r/B onto the same positive radii r_cm by averaging left/right walls
    # (vExB_bin is already signed outward Er/B, so symmetric mean gives the radial speed).
    _, _, vExB_r = antisymmetric_rotation_profile(
        x_mach, vExB_bin, x_center_cm=x_center_cm, axis=0
    )

    shear_mach = radial_flow_shear(
        r_cm, v_rot, smooth_window=smooth_window, polyorder=smooth_poly, axis=0
    )
    shear_dv = radial_flow_shear(
        r_cm, dv_rot, smooth_window=smooth_window, polyorder=smooth_poly, axis=0
    )
    shear_exb = radial_flow_shear(
        r_cm, vExB_r, smooth_window=smooth_window, polyorder=smooth_poly, axis=0
    )

    # Select reference pre-onset Mach bin matched to the paired IV run's ref_sweep_idx
    # (or last finite bias-on bin if ref_sweep_idx is unavailable).
    ref_swp = iv["ref_sweep_idx"]
    hits = np.flatnonzero((swp_idx == ref_swp) & np.any(np.isfinite(vY), axis=0))
    if hits.size > 0:
        ref_bin = int(hits[0])
    else:
        fin_on = np.flatnonzero((t_ms >= 0.0) & np.any(np.isfinite(vY), axis=0))
        ref_bin = int(fin_on[-1]) if fin_on.size > 0 else -1

    # Load optional instability growth rate from onset_summary_jun2026.npz (Day-1 IV run).
    gamma_instab = np.nan
    t_onset_ms = np.nan
    onset_path = onset_npz or os.path.join(data_dir, "onset_summary_jun2026.npz")
    onset_file = ""
    if os.path.isfile(onset_path):
        onset_file = os.path.basename(onset_path)
        with np.load(onset_path) as od:
            if f"{iv_run}__gamma_med_s" in od.files:
                gamma_instab = float(od[f"{iv_run}__gamma_med_s"])
            if f"{iv_run}__t_onset_med_ms" in od.files:
                t_onset_ms = float(od[f"{iv_run}__t_onset_med_ms"])

    out = {
        "mach_run": mach_run,
        "iv_run": iv_run,
        "profiles_file": iv["profiles_file"],
        "onset_file": onset_file,
        "has_pre_bias": has_pre_bias,
        "gas_puff_ms": float(iv["gas_puff_ms"]),
        "x_cm": x_mach,
        "r_cm": r_cm,
        "t_ms": t_ms,
        "iv_sweep_index": swp_idx,
        "Te_bin": Te_bin,
        "Te_err_bin": Te_err_bin,
        "v_ExB_bin_km_s": vExB_bin,
        "v_Y_pre_km_s": vY_pre[:, 0],
        "dv_Y_te": dvY,
        "dv_Y_dM_te": dvY_dM,
        "v_rot_km_s": v_rot,
        "v_sym_km_s": v_sym,
        "dv_rot_km_s": dv_rot,
        "dv_rot_dM_km_s": dv_rot_dM,
        "dv_sym_km_s": dv_sym,
        "v_ExB_r_km_s": vExB_r,
        "dv_dr_mach_s": shear_mach.dv_dr_s,
        "omega_mach_rad_s": shear_mach.omega_rad_s,
        "gamma_E_mach_s": shear_mach.gamma_E_s,
        "gamma_E_dv_s": shear_dv.gamma_E_s,
        "gamma_E_ExB_s": shear_exb.gamma_E_s,
        "ref_bin_idx": ref_bin,
        "ref_bin_t_ms": float(t_ms[ref_bin]) if ref_bin >= 0 else np.nan,
        "ref_iv_swp_idx": int(swp_idx[ref_bin]) if ref_bin >= 0 else -1,
        "t_onset_iv_ms": t_onset_ms,
        "gamma_instab_s": gamma_instab,
        **rescaled,
    }

    if ref_bin >= 0:
        ann = (r_cm >= SHEAR_ANNULUS_R_CM[0]) & (r_cm <= SHEAR_ANNULUS_R_CM[1])
        wall = (r_cm >= INNER_WALL_R_CM[0]) & (r_cm <= INNER_WALL_R_CM[1])

        def _peak_abs(arr, mask):
            vals = arr[mask]
            fin = vals[np.isfinite(vals)]
            return float(fin[int(np.argmax(np.abs(fin)))]) if fin.size > 0 else np.nan

        def _med(arr, mask):
            vals = arr[mask]
            fin = vals[np.isfinite(vals)]
            return float(np.median(fin)) if fin.size > 0 else np.nan

        def _ratio(num, den):
            return (abs(num) / abs(den)
                    if (np.isfinite(num) and np.isfinite(den) and abs(den) > 0)
                    else np.nan)

        # Matched-band comparisons on SHEAR_ANNULUS_R_CM (3-6 cm)
        vrot_ann = _peak_abs(v_rot[:, ref_bin], ann)
        dvrot_ann = _peak_abs(dv_rot[:, ref_bin], ann)
        dvrot_dM_ann = _peak_abs(dv_rot_dM[:, ref_bin], ann)
        vexb_ann = _peak_abs(vExB_r[:, ref_bin], ann)
        ge_mach_ann = _peak_abs(shear_mach.gamma_E_s[:, ref_bin], ann)
        ge_dv_ann = _peak_abs(shear_dv.gamma_E_s[:, ref_bin], ann)
        ge_exb_ann = _peak_abs(shear_exb.gamma_E_s[:, ref_bin], ann)

        # Inner-wall (1.5-4 cm) E x B reference (kept explicitly separate from annulus)
        vexb_wall = _peak_abs(vExB_r[:, ref_bin], wall)
        ge_exb_wall = _peak_abs(shear_exb.gamma_E_s[:, ref_bin], wall)

        out.update({
            "ref_v_rot_ann_km_s": vrot_ann,
            "ref_dv_rot_ann_km_s": dvrot_ann,
            "ref_dv_rot_dM_ann_km_s": dvrot_dM_ann,
            "ref_v_sym_ann_km_s": _med(v_sym[:, ref_bin], ann),
            "ref_v_ExB_ann_km_s": vexb_ann,
            "ref_v_ExB_wall_km_s": vexb_wall,
            "ref_ratio_vrot_to_vExB_ann": _ratio(vrot_ann, vexb_ann),
            "ref_ratio_dvrot_to_vExB_ann": _ratio(dvrot_ann, vexb_ann),
            "ref_ratio_dvrot_dM_to_vExB_ann": _ratio(dvrot_dM_ann, vexb_ann),
            "ref_gamma_E_mach_ann_s": ge_mach_ann,
            "ref_gamma_E_dv_ann_s": ge_dv_ann,
            "ref_gamma_E_ExB_ann_s": ge_exb_ann,
            "ref_gamma_E_ExB_wall_s": ge_exb_wall,
            "ref_ratio_gammaE_to_gamma": _ratio(ge_mach_ann, gamma_instab),
        })
    return out


def _rescaled_settings(x_center_cm, smooth_window, smooth_poly, te_mode):
    return {
        "x_center_cm": float(x_center_cm),
        "smooth_window": int(smooth_window),
        "smooth_poly": int(smooth_poly),
        "te_mode": str(te_mode),
    }


def _rescaled_settings_mismatch(npz_data, run_num, want):
    diffs = []
    for field, val in want.items():
        k = f"{run_num}__{field}"
        if k not in npz_data.files:
            diffs.append(f"{field}: missing in npz (requested {val!r})")
            continue
        if isinstance(val, str):
            got = str(npz_data[k])
            if got != val:
                diffs.append(f"{field}: stored {got!r} != requested {val!r}")
        elif isinstance(val, int):
            got = int(npz_data[k])
            if got != val:
                diffs.append(f"{field}: stored {got} != requested {val}")
        else:
            got = float(npz_data[k])
            if not np.isclose(got, val, atol=1e-9, rtol=0.0):
                diffs.append(f"{field}: stored {got:g} != requested {val:g}")
    return diffs


def batch_rescaled_summary(data_dir=DATA_DIR, runs=None, *, profiles_npz=None,
                           onset_npz=None, out_path=None,
                           x_center_cm=DEFAULT_X_CENTER_CM, smooth_window=5,
                           smooth_poly=2, te_mode="upper", overwrite=False):
    """Batch measured-Te Mach rescaling & ExB shear comparison -> npz path."""
    run_list = sorted(MACH_TO_IV) if runs is None else list(runs)
    out_path = out_path or os.path.join(data_dir, RESCALED_OUT_NAME)
    want = _rescaled_settings(x_center_cm, smooth_window, smooth_poly, te_mode)

    if overwrite and runs is None and os.path.isfile(out_path):
        os.remove(out_path)

    existing_runs = set()
    drop_keys = []
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            existing_runs = {
                k.split("__", 1)[0] for k in d.files if k.endswith("__ref_bin_idx")
            }
            if overwrite:
                prefixes = tuple(f"{r}__" for r in run_list)
                drop_keys = [k for k in d.files if k.startswith(prefixes)]
                existing_runs -= set(run_list)
            else:
                for r in run_list:
                    if r in existing_runs:
                        diffs = _rescaled_settings_mismatch(d, r, want)
                        if diffs:
                            raise ValueError(
                                f"{out_path}: stored run '{r}' differs from "
                                f"requested settings ({'; '.join(diffs)}); "
                                "pass overwrite=True to regenerate")

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    ran_any = False
    for r in run_list:
        if r in existing_runs:
            continue
        res = rescale_line_with_iv(
            r, data_dir=data_dir, profiles_npz=profiles_npz, onset_npz=onset_npz,
            x_center_cm=x_center_cm, smooth_window=smooth_window,
            smooth_poly=smooth_poly, te_mode=te_mode,
        )
        for k, v in res.items():
            if isinstance(v, str):
                payload[f"{r}__{k}"] = np.str_(v)
            elif isinstance(v, (bool, np.bool_)):
                payload[f"{r}__{k}"] = np.bool_(v)
            elif isinstance(v, (int, np.integer)):
                payload[f"{r}__{k}"] = np.int64(v)
            elif isinstance(v, (float, np.floating)):
                payload[f"{r}__{k}"] = np.float64(v)
            else:
                payload[f"{r}__{k}"] = np.asarray(v)
        for sk, sv in want.items():
            if isinstance(sv, str):
                payload[f"{r}__{sk}"] = np.str_(sv)
            elif isinstance(sv, int):
                payload[f"{r}__{sk}"] = np.int64(sv)
            else:
                payload[f"{r}__{sk}"] = np.float64(sv)
        ran_any = True

    if ran_any or not os.path.isfile(out_path):
        stale_keys = [k for k in drop_keys if k not in payload]
        merge_save_npz(out_path, payload, drop_keys=stale_keys)
        print(f"\nWrote {out_path}")
    return out_path


def print_rescaled_summary(npz_path=None):
    """Print matched-band (r in 3-6 cm) measured-Te Mach vs. P29 E x B & shear."""
    npz_path = npz_path or os.path.join(DATA_DIR, RESCALED_OUT_NAME)
    with np.load(npz_path) as d:
        runs = [r for r in sorted(MACH_TO_IV) if f"{r}__ref_bin_idx" in d.files]
        print("\n=== Matched-Band (r in 3-6 cm) Measured-Te Mach (P33) vs. E x B (P29) & Shear ===")
        print(
            f"{'Mach':>4} {'IV':>3} {'Puff':>5} {'t_bin':>6} {'v_rot':>7} {'dv_rot':>7} "
            f"{'dv_dM':>7} {'vExB_a':>7} {'dv/ExB':>7} {'gE_M[/ms]':>10} {'gE_ExB':>8} "
            f"{'g_in':>7} {'|gE|/g':>7}"
        )
        for r in runs:
            ge_m = float(d[f"{r}__ref_gamma_E_mach_ann_s"]) * 1e-3
            ge_e = float(d[f"{r}__ref_gamma_E_ExB_ann_s"]) * 1e-3
            g_in = float(d[f"{r}__gamma_instab_s"]) * 1e-3
            rat_g = float(d[f"{r}__ref_ratio_gammaE_to_gamma"])
            gin_s = f"{g_in:6.2f}" if np.isfinite(g_in) else "   ---"
            rg_s = f"{rat_g:6.1f}x" if np.isfinite(rat_g) else "    ---"
            print(
                f"{r:>4} {str(d[f'{r}__iv_run']):>3} {float(d[f'{r}__gas_puff_ms']):4.0f}m "
                f"{float(d[f'{r}__ref_bin_t_ms']):6.2f} "
                f"{float(d[f'{r}__ref_v_rot_ann_km_s']):+7.2f} "
                f"{float(d[f'{r}__ref_dv_rot_ann_km_s']):+7.2f} "
                f"{float(d[f'{r}__ref_dv_rot_dM_ann_km_s']):+7.2f} "
                f"{float(d[f'{r}__ref_v_ExB_ann_km_s']):+7.2f} "
                f"{float(d[f'{r}__ref_ratio_dvrot_to_vExB_ann']):7.2f} "
                f"{ge_m:+10.1f} {ge_e:+8.1f} {gin_s:>7} {rg_s:>7}"
            )


if __name__ == "__main__":
    batch_all()
