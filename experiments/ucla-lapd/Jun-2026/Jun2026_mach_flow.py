"""Jun-2026 run 32: calibrated Mach-probe flow over the 41x41 P33 plane.

Applies the ``17-20-mach-calibration.npz`` area ratios (``Jun2026_mach_cal``) to
the 3D Mach plane, giving a per-position, per-time-bin flow vector. Run 32 is the
only Jun-2026 run that pairs all six tips with a moving probe and the bias plate
in, so it is the one run where the calibration buys an actual flow map.

Two stages, because the read is the expensive part:

1. :func:`batch_flow` -- reads 1681 positions x 6 tips x 5 shots, reduces each
   shot to :data:`BIN_MS` time bins, and writes a co-located npz. Slow (the whole
   run is ~10 GB of samples); run once.
2. :func:`emit_flow_slider` / :func:`plot_flow_frame` -- read that npz and draw.
   Fast, and re-runnable while choosing a frame or a colour scale.

Geometry
--------
The tip labels are the probe's own axes. X/Y span the scan plane -- the same
plane the probe moves in, so ``M_X``/``M_Y`` are the in-plane vector drawn as
quiver -- and Z is along the machine axis (B), drawn as its own map.

Velocities assume a **constant** electron temperature (:data:`TE_EV`): run 32
digitized no swept tip, so no T_e was measured with this data. ``v = M * c_s``
scales as ``sqrt(T_e)``, so the number is a stated scale, not a measurement; it
is recorded in the npz and printed on every figure. Mach number is the measured
quantity and is stored alongside.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_mach_flow.py
"""

import datetime
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Jun2026_IV as jiv
import Jun2026_plot as jpl
from Jun2026_slider import slider_path
from data_analysis.io import open_lapd, parse_shunts, position_shots
from data_analysis.io.probe_map import channel_wiring
from data_analysis.plasma.mach import (K_HUTCHINSON, face_ratio, flow_velocity,
                                       mach_single, valid_current_mask)
from data_analysis.utils import run_num_of
from data_analysis.viz.plot_utils import (finalize_figure, grid_by_position,
                                          grid_frames, resolve_save)
from data_analysis.viz.slider_html import SCHEMA_VERSION, write_slider_html

DATA_DIR = r"D:\data\LAPD\jun2026-jia"
IFN = os.path.join(DATA_DIR, "32-He-800G-bias40V-Mach-plane_2026-06-13.hdf5")
CAL_NPZ = os.path.join(DATA_DIR, "17-20-mach-calibration.npz")
SCOPE = "scope"

#: Time-bin width [ms]. The run samples at 10 MHz over -2..8 ms, so 0.1 ms is
#: ~1000 samples per bin -- enough to average down shot noise, short enough that
#: the 12-17 ms bias window (0-5 ms in this run's bias-relative frame) resolves
#: into 50 frames rather than a handful.
BIN_MS = 0.1

#: Whole record. t=0 is bias start ("Probe scopes trigger at bias starting
#: time"), so this spans 2 ms of pre-bias baseline, the 5 ms bias-on stretch, and
#: 3 ms of decay -- the baseline is what makes the bias-driven change visible as
#: a change rather than as a level.
WINDOW_MS = (-2.0, 8.0)

#: Assumed electron temperature [eV] and ion mass [proton masses; He = 4].
#: NOT measured in this run -- see the module docstring.
TE_EV = 5.0
ION_MU = 4.0

#: Probe axes, in npz/array order. X and Y span the scan plane (drawn as the
#: quiver); Z is along the machine axis (B) and gets its own map.
AXES = ("X", "Y", "Z")

#: Arrows are drawn every Nth cell in each direction. 41x41 arrows overlap into
#: a solid mat at figure size; 2 is what makes the field readable. The
#: background raster is never decimated.
QUIVER_STEP = 2

#: Window the rotation centre is fitted over [ms]. The bias-on stretch: that is
#: when an E x B rotation exists to have a centre. See :func:`find_flow_centre`.
CENTRE_FIT_MS = (0.0, 5.0)

#: Fewest samples a time bin may hold. A bin is a shot-and-sample pool that one
#: face ratio is formed from, so a handful of samples yields a ratio dominated by
#: whichever sample happened to land in it. Run 32 holds 1000 per bin, far above
#: this -- the floor exists to fail loudly if BIN_MS is ever set near the sample
#: interval, rather than to bind here.
MIN_BIN_SAMPLES = 10

NPZ_SUFFIX = "-mach-flow-data.npz"

# Tip name at the end of a wiring description: 'Isat, Mach@P33-X+' -> 'X+'.
# Same rule as Jun2026_mach_cal: splitting on '-' fails, the sign is a '-'.
_TIP_RE = re.compile(r"([XYZ][+-])\s*$", re.IGNORECASE)


def flow_npz_path(ifn=IFN):
    """Co-located npz beside the raw HDF5, named from the run number."""
    return os.path.join(os.path.dirname(ifn), f"{run_num_of(ifn)}{NPZ_SUFFIX}")


def load_calibration(path=CAL_NPZ):
    """The calibration's ``kappa`` and its systematic bar -> ``(kappa, err_sys)``.

    Both keyed by tip name, so a reordered ``AXES`` here cannot attach one axis'
    area ratio to another. Reads ``kappa_err_sys`` rather than ``kappa_err_fit``:
    for X the fit slot is NaN by construction (X has no fit row), and ``sys`` is
    the bar the calibration script says to quote for every axis.
    """
    with np.load(path) as d:
        tips = [str(t) for t in d["tips"]]
        kappa = {t: float(k) for t, k in zip(tips, d["kappa"])}
        err = {t: float(e) for t, e in zip(tips, d["kappa_err_sys"])}
        method = {t: str(m) for t, m in zip(tips, d["kappa_method"])}
    missing = [a for a in AXES if a not in kappa]
    if missing:
        raise ValueError(f"{path}: calibration has no kappa for {missing}")
    return kappa, err, method


def tip_channels(ifn):
    """``{'X+': 'C1', ...}`` from the run's own wiring descriptions.

    Never a hardcoded C1=X+ table: a swapped +/- assignment inverts R and flips
    the sign of every flow vector, which looks entirely plausible on a map.
    """
    out = {}
    for (_scope, chan), desc in channel_wiring(ifn).items():
        m = _TIP_RE.search(desc)
        if m:
            out[m.group(1).upper()] = chan
    missing = [f"{a}{s}" for a in AXES for s in "+-" if f"{a}{s}" not in out]
    if missing:
        raise ValueError(f"{ifn}: no channel found for {missing}")
    return out


def snap_to_planned(pos):
    """Actual probe coordinates snapped to the planned grid nodes -> ``(x, y)``.

    The drive lands within ~0.07 cm (x) / ~0.15 cm (y) of each node against a
    1.0 cm spacing, so run 32's 1681 positions carry 624 distinct x values.
    Gridding those literally builds a 624x126 mostly-empty raster instead of the
    41x41 plane that was measured; the run log's rule for exactly this is "trust
    the planned grid" (its `positions_setup_array`).

    Verified for run 32 rather than assumed: every position maps to its own node,
    with no node empty and none claimed twice. That is checked here, because a
    silent collision would average two probe locations into one cell.
    """
    px = pos.pos_array["x"][::pos.nshot][:pos.npos]
    py = pos.pos_array["y"][::pos.nshot][:pos.npos]
    ix = np.abs(px[:, None] - pos.xpos[None, :]).argmin(1)
    iy = np.abs(py[:, None] - pos.ypos[None, :]).argmin(1)
    counts = np.zeros((pos.ypos.size, pos.xpos.size), int)
    np.add.at(counts, (iy, ix), 1)
    if (counts > 1).any():
        n = int((counts > 1).sum())
        raise ValueError(
            f"{n} planned node(s) claimed by more than one position; snapping "
            "would merge distinct probe locations into one cell. Grid on the "
            "recorded coordinates instead.")
    return pos.xpos[ix], pos.ypos[iy]


def bin_edges(tarr, window_ms=WINDOW_MS, bin_ms=BIN_MS):
    """Sample-index edges of the time bins -> ``(edges, centres_ms)``.

    ``edges`` has ``nbin+1`` entries indexing ``tarr``; bin ``k`` is
    ``[edges[k], edges[k+1])``. Bins are defined on the *time* axis and then
    located in samples, so a bin is ``bin_ms`` wide by construction even where
    the sample count per bin is not exactly constant.
    """
    t0, t1 = window_ms
    nbin = int(round((t1 - t0) / bin_ms))
    t_edges = t0 + bin_ms * np.arange(nbin + 1)
    edges = np.searchsorted(tarr, t_edges * 1e-3)
    thin = np.flatnonzero(np.diff(edges) < MIN_BIN_SAMPLES)
    if thin.size:
        k = int(thin[0])
        raise ValueError(
            f"bin {k} ({t_edges[k]:g}-{t_edges[k + 1]:g} ms) holds "
            f"{int(edges[k + 1] - edges[k])} sample(s), below the "
            f"{MIN_BIN_SAMPLES} needed for a meaningful ratio; the record spans "
            f"{tarr[0] * 1e3:.2f}-{tarr[-1] * 1e3:.2f} ms at "
            f"{1e-6 / (tarr[1] - tarr[0]):.1f} MHz, so WINDOW_MS reaches outside "
            f"it or BIN_MS is shorter than the sample interval.")
    return edges, t_edges[:-1] + bin_ms / 2


def _binned_face_ratio(plus, minus, edges):
    """Face ratio per time bin from two ``(nshot, nt)`` stacks -> ``(nbin,)``.

    Reduces over both shots and the bin's samples, keeping ``face_ratio``'s
    average-then-divide invariant: the bin is flattened so the pooled mean of
    each face is formed before the division, never a mean of per-sample ratios.
    Also returns the valid-sample count per bin, so an empty bin is visible as a
    count rather than as a plausible NaN.
    """
    nbin = edges.size - 1
    ratio = np.full(nbin, np.nan)
    counts = np.zeros(nbin, dtype=np.int32)
    for k in range(nbin):
        p = plus[:, edges[k]:edges[k + 1]].ravel()
        m = minus[:, edges[k]:edges[k + 1]].ravel()
        ok = valid_current_mask(p, m)
        counts[k] = ok.sum()
        if counts[k]:
            # axis=0 on the flattened pair: one ratio for the whole bin.
            ratio[k] = face_ratio(p, m, axis=0, mask=ok)
    return ratio, counts


def batch_flow(ifn=IFN, cal_path=CAL_NPZ, window_ms=WINDOW_MS, bin_ms=BIN_MS,
               te_ev=TE_EV, out_path=None):
    """Read the plane, reduce to time bins, apply kappa -> co-located npz.

    The slow stage. For every probe position it reads all six tips' shots, bins
    them to ``bin_ms``, forms each axis' face ratio, and converts to a Mach
    number with that axis' calibrated kappa. Only the reduced
    ``(npos, nbin)`` arrays survive the loop -- the raw stacks are ~2.4 MB per
    tip per position and are dropped as soon as they are binned, so peak memory
    is one position's six stacks rather than the run.

    Writes ``<run>-mach-flow-data.npz``: ``M_X``/``M_Y``/``M_Z`` and the matching
    ``v_*`` [km/s], ``n_valid``, the position axes, the bin centres, and the
    calibration and T_e that were applied. Returns the npz path.
    """
    kappa, kappa_err, method = load_calibration(cal_path)
    chans = tip_channels(ifn)
    run = open_lapd(ifn)
    shunts = parse_shunts(run.description().raw)
    no_shunt = [t for t in chans if t not in shunts]
    if no_shunt:
        # Never default: X- is 300 ohm against 75/43 elsewhere, so a guessed
        # value is a 4x gain landing silently in every velocity.
        raise ValueError(f"{ifn}: no shunt in description for {no_shunt}")

    pos = jiv.read_lp_positions(ifn)
    tarr = run.time_array(scope_name=SCOPE)
    edges, t_ms = bin_edges(tarr, window_ms, bin_ms)
    nbin = t_ms.size

    mach = {a: np.full((pos.npos, nbin), np.nan) for a in AXES}
    n_valid = np.zeros((len(AXES), pos.npos, nbin), dtype=np.int32)

    skipped = []
    for p in tqdm(range(pos.npos), desc="Mach flow", unit="pos"):
        shots = position_shots(pos.shot_nums, p, pos.nshot)
        for i, axis in enumerate(AXES):
            stacks = []
            for sign in "+-":
                stack, _ = run.channel(chans[f"{axis}{sign}"],
                                       scope_name=SCOPE, shots=shots)
                # A position the scope never wrote comes back as None. Leave its
                # row NaN and record it: an unwritten position dropped silently
                # is indistinguishable from one that measured zero flow.
                if stack is None:
                    skipped.append((p, f"{axis}{sign}"))
                    stacks = None
                    break
                # volts across the shunt -> amps, per tip (the tips differ:
                # 75/300/75/75/43/43 ohm). In place: the read allocated this
                # 4 MB stack and nothing else refers to it, so an out-of-place
                # divide would copy it 10086 times over the run.
                stack /= shunts[f"{axis}{sign}"]
                stacks.append(stack)
            if stacks is None:
                continue
            R, counts = _binned_face_ratio(stacks[0], stacks[1], edges)
            n_valid[i, p] = counts
            mach[axis][p] = mach_single(R, kappa[axis])

    lost = sorted({p for p, _ in skipped})
    if lost:
        print(f"\n  {len(lost)} position(s) had no scope data and stay NaN: "
              f"{lost[:10]}{' ...' if len(lost) > 10 else ''}")

    pos_x, pos_y = snap_to_planned(pos)

    arrays = {
        "pos_x": pos_x, "pos_y": pos_y, "t_ms": t_ms,
        "n_missing_positions": np.int32(len(lost)),
        # 'axes' is provenance, not indexing state: it lets a reader outside this
        # repo learn the M_*/v_* key order without the module constant.
        "n_valid": n_valid, "axes": np.array(AXES),
        "kappa": np.array([kappa[a] for a in AXES]),
        "kappa_err_sys": np.array([kappa_err[a] for a in AXES]),
        "kappa_method": np.array([method[a] for a in AXES]),
        "shunts": np.array([shunts[f"{a}{s}"] for a in AXES for s in "+-"]),
        "tip_channels": np.array([f"{a}{s}={chans[f'{a}{s}']}"
                                  for a in AXES for s in "+-"]),
        "te_ev": np.float64(te_ev), "ion_mu": np.float64(ION_MU),
        "bin_ms": np.float64(bin_ms), "window_ms": np.array(window_ms),
        "K_hutchinson": np.float64(K_HUTCHINSON),
        "nshot": np.int32(pos.nshot),
        "source_file": np.str_(os.path.basename(ifn)),
        "calibration_file": np.str_(os.path.basename(cal_path)),
        "created": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    for a in AXES:
        arrays[f"M_{a}"] = mach[a]
        arrays[f"v_{a}"] = flow_velocity(mach[a], te_ev, ION_MU)

    out_path = out_path or flow_npz_path(ifn)
    np.savez(out_path, **arrays)
    print(f"\nWrote {out_path} ({pos.npos} positions x {nbin} bins of "
          f"{bin_ms} ms, T_e = {te_ev} eV assumed)")
    return out_path


def load_flow(npz_path=None):
    """The batched arrays as a plain dict, npz handle closed."""
    with np.load(npz_path or flow_npz_path()) as d:
        return {k: d[k] for k in d.files}


def _te_note(data):
    """The one-line T_e caveat every velocity render carries."""
    return (f"v = M x c_s assumes T_e = {float(data['te_ev']):g} eV "
            f"(not measured in this run); v scales as sqrt(T_e)")


def _kappa_note(data):
    """Per-axis kappa and its systematic bar, for the provenance banner."""
    return {f"kappa_{a}": f"{k:.4f} x/÷{e:.3f}  [{m}]"
            for a, k, e, m in zip(AXES, data["kappa"], data["kappa_err_sys"],
                                  data["kappa_method"])}


def _weakest_axis_note(data):
    """Name the axis whose kappa carries the largest systematic bar.

    Read from the npz rather than written as a literal: the numbers come from
    whichever calibration was applied, and a hardcoded "x/÷1.46" would keep
    reading plausibly after a recalibration changed it.
    """
    err = np.asarray(data["kappa_err_sys"], float)
    w = int(np.argmax(err))
    others = ", ".join(f"x/÷{e:.2f} for {a}"
                       for a, e in zip(AXES, err) if a != AXES[w])
    return (f"kappa_{AXES[w]} carries a x/÷{err[w]:.2f} systematic ({others}), "
            f"so the {AXES[w]} component's magnitude is the weakest number here.")


def polar_components(vx, vy, pos_x, pos_y, centre):
    """In-plane flow resolved about ``centre`` -> ``(v_r, v_theta)``, same shape.

    ``v_r`` is positive outward; ``v_theta`` is positive counter-clockwise (the
    +z sense, with z out of the x-y plane along B).

    The decomposition is the point of the measurement, not a presentation
    choice: the biasing plate drives an azimuthal E x B flow, which in Cartesian
    components reverses sign across the column and averages to nearly zero over
    a symmetric plane -- structure that a v_X/v_Y map shows only as a pattern the
    reader must integrate by eye. ``v_theta`` states it as one number per cell.

    ``centre`` is in the *same machine coordinates* as ``pos_x``/``pos_y``; only
    the projection uses it. Nothing is translated, so every plot keeps machine
    coordinates and a feature at x = 5 cm stays at x = 5 cm.
    """
    dx = np.asarray(pos_x, float) - centre[0]
    dy = np.asarray(pos_y, float) - centre[1]
    r = np.hypot(dx, dy)
    # r == 0 has no defined direction; one cell at most, left NaN rather than
    # given an arbitrary unit vector.
    with np.errstate(invalid="ignore", divide="ignore"):
        ur, ut = np.where(r > 0, dx / r, np.nan), np.where(r > 0, dy / r, np.nan)
    # Broadcast the per-position unit vectors against (npos, nbin) cubes.
    if vx.ndim == 2:
        ur, ut = ur[:, None], ut[:, None]
    return vx * ur + vy * ut, -vx * ut + vy * ur


def find_flow_centre(data, window_ms=CENTRE_FIT_MS, search_cm=3.0, step_cm=0.25,
                     r_min=3.0, r_max=15.0):
    """The rotation centre, fitted from the flow field itself -> ``(cx, cy)`` cm.

    A rigid rotation has one stagnation point, and about the true centre the flow
    is purely azimuthal. This scans candidate centres and takes the one
    minimising ``mean|v_r| / mean|v_theta|`` over the annulus ``r_min..r_max``,
    averaged over ``window_ms``.

    Fitted rather than taken from the run log's "plate centred ~(0,0)": the log's
    own tilde is the point -- the plate position is approximate, and v_r/v_theta
    are sensitive to the centre in a way the Cartesian components are not.

    The annulus matters. Including r < r_min lets the near-stagnant core, where
    the direction is ill-defined, dominate the ratio; including the plane edge
    lets uniformly low-signal cells win by having no flow to be radial. Both were
    measured to move the answer.

    Measured for run 32: (+2.5, -1.5) during bias, ratio 0.22, stable across the
    bias window. Outside it the fit degenerates (the ratio rises to ~0.7 and the
    best centre runs to the search-box edge) -- correctly, since there is no
    rotation to centre when the plate is off.
    """
    t = data["t_ms"]
    win = (t >= window_ms[0]) & (t <= window_ms[1])
    if not win.any():
        raise ValueError(f"centre-fit window {window_ms} ms selects no bin of "
                         f"{t[0]:.2f}..{t[-1]:.2f} ms")
    # An all-NaN row is a position the scope never wrote (see batch_flow); it
    # comes back NaN and is excluded by the isfinite masks below.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        vx = np.nanmean(data["v_X"][:, win], axis=1)
        vy = np.nanmean(data["v_Y"][:, win], axis=1)
    px, py = data["pos_x"], data["pos_y"]

    grid = np.arange(-search_cm, search_cm + step_cm / 2, step_cm)
    best = None
    for cx in grid:
        for cy in grid:
            r = np.hypot(px - cx, py - cy)
            keep = (r > r_min) & (r < r_max) & np.isfinite(vx) & np.isfinite(vy)
            if keep.sum() < 50:
                continue
            vr, vt = polar_components(vx[keep], vy[keep], px[keep], py[keep],
                                      (cx, cy))
            ratio = np.mean(np.abs(vr)) / (np.mean(np.abs(vt)) + 1e-12)
            if best is None or ratio < best[0]:
                best = (ratio, float(cx), float(cy))
    if best is None:
        raise ValueError("no candidate centre had enough valid cells in the "
                         f"annulus {r_min}-{r_max} cm")
    return best[1], best[2], best[0]


def emit_flow_slider(npz_path=None, out=None, quiver_step=QUIVER_STEP,
                     vmax=None):
    """Time-slider page: v_X/v_Y quiver over |v|, beside the v_Z map.

    Two **panels**, not two dropdown groups: the axial flow is a different
    quantity on a different (diverging, signed) colour scale, not another
    version of the in-plane one. Panels share the single time slider, so the
    in-plane and axial flow are read at the same instant instead of by switching
    -- which is the comparison a 3D probe exists to support.

    ``vmax`` fixes both colour scales [km/s]; ``None`` lets the page autoscale
    per frame, which is right while hunting for structure and wrong when
    comparing frames.
    """
    data = load_flow(npz_path)
    t_ms = data["t_ms"]

    vx, vy, vz = (data[f"v_{a}"] for a in AXES)
    cx, cy, ratio = find_flow_centre(data)
    v_r, v_th = polar_components(vx, vy, data["pos_x"], data["pos_y"], (cx, cy))

    grid = lambda a: grid_frames(data["pos_x"], data["pos_y"], a)[0]
    fx, xs, ys = grid_frames(data["pos_x"], data["pos_y"], vx)
    fy, fs, fz = grid(vy), grid(np.hypot(vx, vy)), grid(vz)
    f_th, f_r = grid(v_th), grid(v_r)

    # Both scales are fixed or both autoscale, matching the schema's own rule
    # that vmin and vmax are set together.
    fixed = vmax is not None
    fields = [
        {"name": "in-plane flow (v_X, v_Y)", "unit": "km/s", "frames": fs,
         "cmap": "viridis", "vmin": 0.0 if fixed else None, "vmax": vmax,
         # Arrows over the speed map: direction and magnitude of one flow, on
         # one panel and one slider.
         "vectors": {"u": fx, "v": fy, "step": quiver_step}},
        # The E x B panel. Diverging, because the sign is the physics: one sign
        # over the whole ring is a coherent rotation, and a sign that flips
        # across the column is not.
        {"name": "azimuthal flow (v_theta, +ve CCW)", "unit": "km/s",
         "frames": f_th, "cmap": "RdBu_r",
         "vmin": -vmax if fixed else None, "vmax": vmax},
        # Beside it because it is the control: an E x B rotation has little
        # radial flow, so v_r is how a reader judges whether the v_theta panel
        # is a rotation or just a decomposition of something else.
        {"name": "radial flow (v_r, +ve outward)", "unit": "km/s",
         "frames": f_r, "cmap": "RdBu_r",
         "vmin": -vmax if fixed else None, "vmax": vmax},
        {"name": "axial flow (v_Z, along B)", "unit": "km/s",
         # Diverging and symmetric: v_Z is signed, and a sequential map would
         # hide which way the axial flow points.
         "frames": fz, "cmap": "RdBu_r",
         "vmin": -vmax if fixed else None, "vmax": vmax},
    ]

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": f"Run {run_num_of(str(data['source_file']))} - "
                 f"calibrated Mach flow, P33 plane",
        "geometry": "plane",
        "axis": {"name": "time", "unit": "ms", "values": t_ms},
        "x": {"label": "X position", "unit": "cm", "values": xs},
        "y": {"label": "Y position", "unit": "cm", "values": ys},
        "fields": fields,
        "provenance": {
            "source": str(data["source_file"]),
            "params": {"bin": f"{float(data['bin_ms']):g} ms",
                       "shots/position": int(data["nshot"]),
                       "T_e assumed": f"{float(data['te_ev']):g} eV",
                       "calibration": str(data["calibration_file"]),
                       "arrow decimation": f"every {quiver_step} cells",
                       # Coordinates are NOT shifted -- only the v_r/v_theta
                       # projection uses this point.
                       "v_r/v_theta centre": f"({cx:+.2f}, {cy:+.2f}) cm, "
                                             f"fitted over "
                                             f"{CENTRE_FIT_MS[0]:g}-{CENTRE_FIT_MS[1]:g} ms "
                                             f"(|v_r|/|v_theta| = {ratio:.2f})"},
            "details": _kappa_note(data),
        },
        "warning": _te_note(data) + ". " + _weakest_axis_note(data),
    }
    name = f"{run_num_of(str(data['source_file']))}-mach-flow-slider"
    return write_slider_html(bundle, out or slider_path(name))


def plot_flow_frame(t_ms, npz_path=None, quiver_step=QUIVER_STEP, vmax=None,
                    save_fig=True):
    """Static figure at the bin nearest ``t_ms``: in-plane, v_theta, v_r, v_Z.

    The publication counterpart of the slider: scrub the page to find the frame,
    then draw it here. Carries the same four panels in the same order, so a
    figure and the page it came from cannot show different things. ``save_fig``
    follows the campaign driver convention
    (:func:`data_analysis.viz.plot_utils.resolve_save`): ``True`` -> the
    centralized figure path, a string -> that path, falsey -> don't save.
    """
    data = load_flow(npz_path)
    k = int(np.argmin(np.abs(data["t_ms"] - t_ms)))
    cx, cy, ratio = find_flow_centre(data)
    v_r, v_th = polar_components(data["v_X"][:, k], data["v_Y"][:, k],
                                 data["pos_x"], data["pos_y"], (cx, cy))
    # grid_by_position takes one value per position and returns the imshow
    # extent as cell EDGES; building it from the axis vectors instead would put
    # the limits at cell centres, shrinking the map by half a cell each side.
    grid = lambda v: grid_by_position(data["pos_x"], data["pos_y"], v)
    vx, extent = grid(data["v_X"][:, k])
    vy, _ = grid(data["v_Y"][:, k])
    vz, _ = grid(data["v_Z"][:, k])
    g_th, _ = grid(v_th)
    g_r, _ = grid(v_r)
    speed = np.hypot(vx, vy)
    peak = np.nanmax(speed)

    fig, axs = plt.subplots(1, 4, figsize=(22, 5.4), sharey=True)
    ax_p, ax_th, ax_r, ax_z = axs

    im = ax_p.imshow(speed, origin="lower", extent=extent, cmap="viridis",
                     vmin=None if vmax is None else 0.0, vmax=vmax,
                     interpolation="nearest")
    s = quiver_step
    xs = np.linspace(extent[0], extent[1], vx.shape[1], endpoint=False)
    ys = np.linspace(extent[2], extent[3], vx.shape[0], endpoint=False)
    # Cell centres: extent gives edges, and an arrow belongs on the position it
    # was measured at, not on the corner of its cell.
    xs += (xs[1] - xs[0]) / 2 if xs.size > 1 else 0
    ys += (ys[1] - ys[0]) / 2 if ys.size > 1 else 0
    X, Y = np.meshgrid(xs[::s], ys[::s])
    q = ax_p.quiver(X, Y, vx[::s, ::s], vy[::s, ::s], color="w",
                    pivot="mid", scale_units="xy")
    ax_p.quiverkey(q, 0.88, 1.03, peak or 1.0, f"{peak:.1f} km/s",
                   labelpos="E", color="k")
    ax_p.set_title("in-plane flow  (v_X, v_Y)")
    ax_p.set_ylabel("Y [cm]")
    fig.colorbar(im, ax=ax_p, label="|v| in-plane [km/s]")

    for ax, g, title, label in (
            (ax_th, g_th, r"azimuthal  ($v_\theta$, +ve CCW)", r"$v_\theta$ [km/s]"),
            (ax_r, g_r, r"radial  ($v_r$, +ve outward)", r"$v_r$ [km/s]"),
            (ax_z, vz, "axial  ($v_Z$, along B)", r"$v_Z$ [km/s]")):
        lim = vmax or np.nanmax(np.abs(g))
        img = ax.imshow(g, origin="lower", extent=extent, cmap="RdBu_r",
                        vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(img, ax=ax, label=label)

    # The fitted centre, marked on the two panels resolved about it. Machine
    # coordinates throughout -- the marker moves, the axes do not.
    for ax in (ax_th, ax_r):
        ax.plot(cx, cy, "k+", ms=11, mew=1.6)
    for ax in axs:
        ax.set_xlabel("X [cm]")

    run_num = run_num_of(str(data["source_file"]))
    fig.suptitle(f"Run {run_num}  -  t = {data['t_ms'][k]:+.2f} ms "
                 f"(t=0 bias start)  -  {_te_note(data)}  -  "
                 f"polar centre ({cx:+.2f}, {cy:+.2f}) cm", fontsize=9)
    name = f"{run_num}-mach-flow-{data['t_ms'][k]:+.2f}ms"
    finalize_figure(fig, save_fig=resolve_save(save_fig, name, jpl.FIG_SUBDIR))


if __name__ == "__main__":
    if not os.path.exists(flow_npz_path()):
        batch_flow()
    emit_flow_slider()
