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
Tip labels are the probe's own axes: X/Y span the scan plane (the in-plane vector
drawn as quiver), Z is along the machine axis (B) and gets its own map.

Velocities assume a **constant** T_e (:data:`TE_EV`) -- run 32 digitized no swept
tip, so none was measured. ``v = M * c_s`` scales as ``sqrt(T_e)``, making it a
stated scale, not a measurement; it is recorded in the npz and printed on every
figure. Mach number is the measured quantity and is stored alongside.

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
from data_analysis.plasma.mach import (K_HUTCHINSON, binned_face_ratio,
                                       find_flow_centre as _fit_centre,
                                       flow_velocity, mach_single,
                                       polar_components, time_bin_edges)
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


def batch_flow(ifn=IFN, cal_path=CAL_NPZ, window_ms=WINDOW_MS, bin_ms=BIN_MS,
               te_ev=TE_EV, out_path=None):
    """Read the plane, reduce to time bins, apply kappa -> co-located npz.

    The slow stage. Per position: read all six tips' shots, bin to ``bin_ms``,
    form each axis' face ratio, convert with that axis' kappa. Only the reduced
    ``(npos, nbin)`` arrays survive the loop -- raw stacks are ~2.4 MB per tip per
    position and are dropped once binned, so peak memory is one position's six
    stacks rather than the run.

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
        # Never default: the tips differ (300 ohm on X-, 75/43 elsewhere), so a
        # guessed value yields a wrong current with nothing downstream complaining.
        raise ValueError(f"{ifn}: no shunt in description for {no_shunt}")

    pos = jiv.read_lp_positions(ifn)
    tarr = run.time_array(scope_name=SCOPE)
    edges, t_ms = time_bin_edges(tarr, window_ms, bin_ms)
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
            # Amplitude is unused here: run 32 has no decay flag (see the
            # Mar-2026 script, which does). Named rather than indexed so the
            # omission is visible as a choice.
            R, counts, _amplitude = binned_face_ratio(stacks[0], stacks[1],
                                                      edges)
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
    """Per-axis kappa and its systematic bar, for the provenance banner.

    Axis names come from the npz, not the module constant: these strings label a
    banner nobody re-checks, so a reordered ``AXES`` would silently attach one
    axis' kappa to another's name.
    """
    return {f"kappa_{a}": f"{k:.4f} x/÷{e:.3f}  [{m}]"
            for a, k, e, m in zip(data["axes"], data["kappa"],
                                  data["kappa_err_sys"], data["kappa_method"])}


def _weakest_axis_note(data):
    """Name the axis whose kappa carries the largest systematic bar.

    Read from the npz, never a literal: a hardcoded "x/÷1.46" would keep reading
    plausibly after a recalibration changed it. Same for the axis names -- see
    :func:`_kappa_note`.
    """
    axes = [str(a) for a in data["axes"]]
    err = np.asarray(data["kappa_err_sys"], float)
    w = int(np.argmax(err))
    others = ", ".join(f"x/÷{e:.2f} for {a}"
                       for a, e in zip(axes, err) if a != axes[w])
    return (f"kappa_{axes[w]} carries a x/÷{err[w]:.2f} systematic ({others}), "
            f"so the {axes[w]} component's magnitude is the weakest number here.")


def find_flow_centre(data, window_ms=CENTRE_FIT_MS):
    """The rotation centre over ``window_ms`` -> ``(cx, cy, ratio)`` cm.

    Campaign wrapper over :func:`~data_analysis.plasma.mach.find_flow_centre`:
    picks the bins to average and hands it one velocity per position. The window
    is the bias-on stretch -- that is when an E x B rotation exists to have a
    centre at all.

    Run 32: (+2.50, -1.25) during bias, ratio 0.22, stable across the window.
    Outside it the fit degenerates (ratio ~0.7, centre at the search-box edge) --
    correctly, since there is no rotation to centre when the plate is off.
    """
    t = data["t_ms"]
    win = (t >= window_ms[0]) & (t <= window_ms[1])
    if not win.any():
        raise ValueError(f"centre-fit window {window_ms} ms selects no bin of "
                         f"{t[0]:.2f}..{t[-1]:.2f} ms")
    # An all-NaN row is a position the scope never wrote (see batch_flow); it
    # comes back NaN and the fit excludes it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        vx = np.nanmean(data["v_X"][:, win], axis=1)
        vy = np.nanmean(data["v_Y"][:, win], axis=1)
    return _fit_centre(vx, vy, data["pos_x"], data["pos_y"])


def emit_flow_slider(npz_path=None, out=None, quiver_step=QUIVER_STEP,
                     vmax=None):
    """Time-slider page: v_X/v_Y quiver over |v|, beside the v_Z map.

    **Panels, not dropdown groups**: they share one time slider, so every
    component is read at the same instant rather than by switching.

    ``vmax`` fixes all colour scales [km/s]; ``None`` autoscales per frame --
    right while hunting for structure, wrong when comparing frames.
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

    # vmin/vmax are set together (schema rule). Speed is unsigned -> sequential
    # from 0; the three signed components are diverging about 0, because for them
    # the sign is the physics -- one sign over the ring is a coherent rotation.
    # v_r is the control: an E x B rotation has little radial flow.
    fixed = vmax is not None
    fields = [
        {"name": "in-plane flow (v_X, v_Y)", "unit": "km/s", "frames": fs,
         "cmap": "viridis", "vmin": 0.0 if fixed else None, "vmax": vmax,
         "vectors": {"u": fx, "v": fy, "step": quiver_step}},
        {"name": "azimuthal flow (v_theta, +ve CCW)", "unit": "km/s",
         "frames": f_th, "cmap": "RdBu_r",
         "vmin": -vmax if fixed else None, "vmax": vmax},
        {"name": "radial flow (v_r, +ve outward)", "unit": "km/s",
         "frames": f_r, "cmap": "RdBu_r",
         "vmin": -vmax if fixed else None, "vmax": vmax},
        {"name": "axial flow (v_Z, along B)", "unit": "km/s",
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
