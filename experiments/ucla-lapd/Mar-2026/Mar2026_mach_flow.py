"""Mar-2026 run 054: calibrated Mach-probe flow over the 21x21 P29 plane.

Runs 055/056 rotate this probe 180 deg at (0,0), bias off, same day and port, so
both axes' kappa come from a rotation pair rather than an assumption. They
replaced the Jun-2026 calibration, whose kappa_X rested on a no-x-flow assumption
(1.9077 against the rotation pair's 0.833) and put a spurious uniform -0.19 on
every M_X -- the constant pre-bias offset the maps showed.

Run 054 recorded no Z current, so there is no v_Z panel -- not because it is zero.

    .venv/Scripts/python.exe experiments/ucla-lapd/Mar-2026/Mar2026_mach_flow.py
"""

import datetime
import os
import re

import numpy as np
from tqdm import tqdm

from data_analysis.io import open_lapd, parse_shunts
from data_analysis.io.probe_map import channel_wiring
from data_analysis.plasma.flow import (find_flow_centre as _fit_centre,
                                       polar_components)
from data_analysis.plasma.mach import (K_HUTCHINSON, binned_face_ratio,
                                       flow_velocity, mach_single,
                                       time_bin_edges)
from data_analysis.utils import run_num_of
from data_analysis.viz.flow_maps import (plot_flow_frame as _draw_flow_frame,
                                         write_flow_slider)
from data_analysis.viz.plot_utils import (fig_path, finalize_figure,
                                          grid_frames, resolve_save)

DATA_DIR = r"D:\data\LAPD\Mar26-data"
IFN = os.path.join(
    DATA_DIR,
    "054-mach-4tip-p29-xycoarse-varbias-p30Vmax 2026-03-06 14.42.54.hdf5")
CAL_NPZ = os.path.join(DATA_DIR, "055-056-mach-calibration.npz")
FIG_SUBDIR = "Mar-2026"

#: Time-bin width [ms]. ~625 samples per bin at this run's 0.16 us sample --
#: averages down shot noise (only 4 shots/position) and still resolves the bias
#: window into 50 frames.
BIN_MS = 0.1

#: Description's "DAQ trigger: 4.5 ms" after the discharge; the stored time axis
#: starts here.
DAQ_TRIGGER_MS = 4.5

#: Description's "Multi-electrode bias: 7.5-12.5 ms".
BIAS_MACHINE_MS = (7.5, 12.5)

#: T_e was NOT measured -- run 054 digitized no swept tip. v = M * c_s scales as
#: sqrt(T_e), making every velocity a stated scale, not a measurement.
TE_EV = 5.0
ION_MU = 4.0

AXES = ("X", "Y")

#: 1 = every arrow: the 21x21 grid is coarse enough not to overlap.
QUIVER_STEP = 1

#: Pre-bias window [ms, machine]: inside "Plasma: 0-10 ms", before the bias, with
#: a margin so neither the DAQ trigger nor the turn-on transient is averaged in.
PREBIAS_MACHINE_MS = (4.6, 7.4)

#: A bin is unreliable once summed tip current falls below this fraction of peak.
#: A Mach number is a RATIO, so it stays finite and smooth as both faces decay
#: into noise -- run 054 drifts to M_X = -0.43 long after the plasma is gone,
#: which reads as strengthening flow rather than an empty machine.
MIN_SIGNAL_FRAC = 0.1

NPZ_SUFFIX = "-mach-flow-data.npz"

# Mar-2026 labels a tip 'Vx+_p29' on the digitizer channel; Jun-2026 wrote
# 'MP@P33, Isat-X+'. Matching the axis letter and sign separately avoids
# splitting on '-', which is also the sign character.
_TIP_RE = re.compile(r"\bV([XY])([+-])", re.IGNORECASE)


def flow_npz_path(ifn=IFN):
    """Co-located npz beside the raw HDF5, named from the run number."""
    return os.path.join(os.path.dirname(ifn), f"{run_num_of(ifn)}{NPZ_SUFFIX}")


def machine_ms(t_ms):
    """Stored time [ms, from DAQ trigger] -> machine time [ms, from discharge].

    Plots name machine time because the run log's timing block is written that
    way; the npz stores the raw axis, so no consumer inherits a silent shift.
    """
    return np.asarray(t_ms, float) + DAQ_TRIGGER_MS


def load_calibration(path=CAL_NPZ):
    """``(kappa, err, method)``, each keyed by tip name so a reordered
    :data:`AXES` cannot attach one axis' area ratio to another.

    Uses ``kappa_err_sys``, not ``kappa_err_fit``: X has no fit row (NaN by
    construction), and sys is the bar to quote for every axis.
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
    """``{'X+': (board, chan), ...}`` from the run's own wiring descriptions.

    Never a hardcoded table: a swapped +/- inverts R and flips the sign of every
    flow vector, which looks entirely plausible on a map.
    """
    out = {}
    for (_adc, chan), desc in channel_wiring(ifn).items():
        m = _TIP_RE.search(desc)
        if m:
            out[f"{m.group(1).upper()}{m.group(2)}"] = chan
    missing = [f"{a}{s}" for a in AXES for s in "+-" if f"{a}{s}" not in out]
    if missing:
        raise ValueError(f"{ifn}: no digitizer channel found for {missing}")
    return out


def read_positions(ifn=IFN):
    """The scan grid. ``pos_x``/``pos_y`` are one coordinate per position;
    ``shot_nums`` is position-major (entry ``p*nshot + k``).

    Coordinates already land on exact nodes (checked: max snap 0.0 cm), so the
    collision check is a guard, not a snap: a collision would average two probe
    locations into one cell.
    """
    with open_lapd(ifn).session() as sess:
        pos_dict, xpos, ypos, _z, npos, nshot = sess.positions()
    pos_array = pos_dict[list(pos_dict)[0]]
    px = pos_array["x"][::nshot][:npos]
    py = pos_array["y"][::nshot][:npos]
    shot_nums = pos_array["shotnum"]

    ix = np.abs(px[:, None] - xpos[None, :]).argmin(1)
    iy = np.abs(py[:, None] - ypos[None, :]).argmin(1)
    counts = np.zeros((ypos.size, xpos.size), int)
    np.add.at(counts, (iy, ix), 1)
    if (counts > 1).any():
        raise ValueError(
            f"{int((counts > 1).sum())} grid node(s) claimed by more than one "
            "position; gridding would merge distinct probe locations into one "
            "cell.")
    return xpos[ix], ypos[iy], xpos, ypos, npos, nshot, shot_nums


def batch_flow(ifn=IFN, cal_path=CAL_NPZ, bin_ms=BIN_MS, te_ev=TE_EV,
               out_path=None):
    """Read the plane, reduce to time bins, apply kappa -> co-located npz path.

    The slow stage; run once. Only the reduced ``(npos, nbin)`` arrays survive
    the loop, so peak memory is one position's four stacks (~1.7 MB each), not
    the 1.25 GB run. The time window comes from the file's own axis rather than
    being declared, so nothing is trimmed on an assumption.
    """
    kappa, kappa_err, method = load_calibration(cal_path)
    chans = tip_channels(ifn)
    run = open_lapd(ifn)

    shunts = parse_shunts(run.description().raw, tuple(chans))
    no_shunt = [t for t in chans if t not in shunts]
    if no_shunt:
        raise ValueError(f"{ifn}: no shunt in description for {no_shunt}")

    pos_x, pos_y, _xs, _ys, npos, nshot, shot_nums = read_positions(ifn)

    skipped = []

    # One held handle: LapdRun.channel() reopens the file per call, which for
    # 441 positions x 4 tips of a 1.25 GB file is the whole cost of the run.
    with run.session() as sess:
        adc, _digi = sess.digitizer_config()
        _d, tarr = sess.read_data(*chans[f"{AXES[0]}+"], index_arr=slice(0, 1),
                                  adc=adc)
        edges, t_ms = time_bin_edges(tarr, (tarr[0] * 1e3, tarr[-1] * 1e3),
                                     bin_ms)
        nbin = t_ms.size
        mach = {a: np.full((npos, nbin), np.nan) for a in AXES}
        n_valid = np.zeros((len(AXES), npos, nbin), dtype=np.int32)
        # Summed tip current per bin [A]: the amplitude a ratio throws away, and
        # the only thing that distinguishes real flow from two decayed faces.
        signal = np.zeros((npos, nbin))
        n_axes_read = np.zeros(npos, dtype=np.int32)

        for p in tqdm(range(npos), desc="Mach flow", unit="pos"):
            # Rows are position-major and blocked by nshot; shot numbers are
            # contiguous 1..N here (verified in read_positions), so the row
            # slice and the recorded shot numbers agree.
            rows = slice(p * nshot, (p + 1) * nshot)
            for i, axis in enumerate(AXES):
                stacks = []
                for sign in "+-":
                    board, chan = chans[f"{axis}{sign}"]
                    data, _t = sess.read_data(board, chan, index_arr=rows,
                                              adc=adc)
                    stack = np.asarray(data["signal"], dtype=float)
                    if stack.size == 0:
                        skipped.append((p, f"{axis}{sign}"))
                        stacks = None
                        break
                    stack /= shunts[f"{axis}{sign}"]
                    stacks.append(stack)
                if stacks is None:
                    continue
                R, counts, amplitude = binned_face_ratio(stacks[0], stacks[1],
                                                         edges)
                n_valid[i, p] = counts
                mach[axis][p] = mach_single(R, kappa[axis])
                signal[p] += amplitude
                n_axes_read[p] += 1

    lost = sorted({p for p, _ in skipped})
    if lost:
        print(f"\n  {len(lost)} position(s) had no digitizer data and stay NaN: "
              f"{lost[:10]}{' ...' if len(lost) > 10 else ''}")

    complete = n_axes_read == len(AXES)
    if not complete.any():
        raise ValueError(f"{ifn}: no position read all of {list(AXES)}; the "
                         "decay curve has nothing complete to average.")
    decay = signal[complete].mean(axis=0)
    above = decay >= MIN_SIGNAL_FRAC * np.nanmax(decay)
    # Everything after the first drop below the floor is unreliable, even if the
    # noise wanders back above it later.
    reliable = np.logical_and.accumulate(above)

    arrays = {
        "pos_x": pos_x, "pos_y": pos_y, "t_ms": t_ms,
        "n_missing_positions": np.int32(len(lost)),
        # signal_decay_a is NOT signal_a.mean(0): it averages only the positions
        # that read every axis, so it cannot be re-derived from signal_a alone.
        "signal_a": signal, "signal_decay_a": decay, "reliable": reliable,
        "n_axes_read": n_axes_read,
        "min_signal_frac": np.float64(MIN_SIGNAL_FRAC),
        "n_valid": n_valid, "axes": np.array(AXES),
        "kappa": np.array([kappa[a] for a in AXES]),
        "kappa_err_sys": np.array([kappa_err[a] for a in AXES]),
        "kappa_method": np.array([method[a] for a in AXES]),
        "shunts": np.array([shunts[f"{a}{s}"] for a in AXES for s in "+-"]),
        "tip_channels": np.array([f"{a}{s}=board{chans[f'{a}{s}'][0]}"
                                  f"ch{chans[f'{a}{s}'][1]}"
                                  for a in AXES for s in "+-"]),
        "te_ev": np.float64(te_ev), "ion_mu": np.float64(ION_MU),
        "bin_ms": np.float64(bin_ms),
        # The stored axis is w.r.t. the DAQ trigger; this is what converts it to
        # machine time. Stored, not applied -- see machine_ms.
        "daq_trigger_ms": np.float64(DAQ_TRIGGER_MS),
        "bias_machine_ms": np.array(BIAS_MACHINE_MS),
        "K_hutchinson": np.float64(K_HUTCHINSON),
        "nshot": np.int32(nshot),
        "source_file": np.str_(os.path.basename(ifn)),
        "calibration_file": np.str_(os.path.basename(cal_path)),
        "created": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    for a in AXES:
        arrays[f"M_{a}"] = mach[a]
        arrays[f"v_{a}"] = flow_velocity(mach[a], te_ev, ION_MU)

    out_path = out_path or flow_npz_path(ifn)
    np.savez(out_path, **arrays)
    print(f"\nWrote {out_path} ({npos} positions x {nbin} bins of {bin_ms} ms, "
          f"{t_ms[0]:.2f}-{t_ms[-1]:.2f} ms stored / "
          f"{machine_ms(t_ms[0]):.2f}-{machine_ms(t_ms[-1]):.2f} ms machine, "
          f"T_e = {te_ev} eV assumed)")
    n_ok = int(reliable.sum())
    print(f"  signal peaks at {np.nanmax(decay) * 1e3:.2f} mA summed; "
          f"{n_ok}/{nbin} bins stay above {MIN_SIGNAL_FRAC:.0%} of it "
          f"(through {machine_ms(t_ms[n_ok - 1]):.2f} ms machine). "
          f"Later bins are computed but flagged unreliable.")
    return out_path


def load_flow(npz_path=None):
    """The batched arrays as a plain dict, npz handle closed."""
    with np.load(npz_path or flow_npz_path()) as d:
        return {k: d[k] for k in d.files}


def _te_note(data):
    """The one-line T_e caveat every velocity render carries."""
    return (f"v = M x c_s assumes T_e = {float(data['te_ev']):g} eV "
            f"(not measured in this run); v scales as sqrt(T_e)")


def find_flow_centre(data, window_ms=None):
    """The rotation centre over the bias window -> ``(cx, cy, ratio)`` cm.

    ``window_ms`` is *stored* time; the default is the bias window, which is when
    an E x B rotation exists to have a centre at all. ``ratio`` is the fit's own
    quality flag -- small means well-centred, ~1 means there was no rotation.
    """
    if window_ms is None:
        offset = float(data["daq_trigger_ms"])
        b0, b1 = data["bias_machine_ms"]
        window_ms = (b0 - offset, b1 - offset)
    t = data["t_ms"]
    # Excluded from the fit, not just the plot: decayed-noise M would drag the
    # centre toward whatever the noise favoured.
    win = (t >= window_ms[0]) & (t <= window_ms[1]) & data["reliable"]
    if not win.any():
        raise ValueError(
            f"centre-fit window {window_ms} ms (stored) selects no reliable bin "
            f"of {t[0]:.2f}..{t[-1]:.2f} ms; the plasma signal is above the "
            f"{float(data['min_signal_frac']):.0%} floor only through "
            f"{t[int(data['reliable'].sum()) - 1]:.2f} ms stored.")
    vx = np.nanmean(data["v_X"][:, win], axis=1)
    vy = np.nanmean(data["v_Y"][:, win], axis=1)
    return _fit_centre(vx, vy, data["pos_x"], data["pos_y"])


def prebias_offset(data, window_ms=PREBIAS_MACHINE_MS):
    """Uniform (v_X, v_Y) offset [km/s] averaged over the no-bias window.

    NOT known to be instrumental: a real uniform flow with the bias off would
    look identical.
    """
    t = machine_ms(data["t_ms"])
    win = (t >= window_ms[0]) & (t <= window_ms[1]) & data["reliable"]
    if not win.any():
        raise ValueError(
            f"pre-bias window {window_ms} ms (machine) selects no reliable bin "
            f"of {t[0]:.2f}..{t[-1]:.2f} ms machine.")
    return (float(np.nanmean(data["v_X"][:, win])),
            float(np.nanmean(data["v_Y"][:, win])))


def emit_flow_slider(npz_path=None, out=None, quiver_step=QUIVER_STEP,
                     vmax=None, subtract_offset=False):
    """Time-slider page: v_X/v_Y quiver over |v|, plus v_theta and v_r.

    All components are panels on one time slider.
    """
    data = load_flow(npz_path)
    vx, vy = data["v_X"], data["v_Y"]

    off = None
    if subtract_offset:
        # Subtract an offset that is uniform over the plane, averaged over the pre-bias window.
        off = prebias_offset(data)
        vx, vy = vx - off[0], vy - off[1]
        data = dict(data, v_X=vx, v_Y=vy)

    cx, cy, ratio = find_flow_centre(data)
    v_r, v_th = polar_components(vx, vy, data["pos_x"], data["pos_y"], (cx, cy))

    bias = np.asarray(data["bias_machine_ms"], float)

    # Ion saturation current itself, as a field: it is the amplitude the Mach
    # ratio divides away, so it is what tells a reader whether a frame shows
    # flow in a plasma or a ratio of two decayed traces.
    amp = grid_frames(data["pos_x"], data["pos_y"], data["signal_a"] * 1e3)[0]
    extra = [{"name": "summed tip current (signal amplitude)", "unit": "mA",
              "frames": amp, "cmap": "magma",
              "vmin": 0.0 if vmax is not None else None, "vmax": None}]

    run = run_num_of(str(data["source_file"]))
    name = f"{run}-mach-flow-slider" + ("-offset-removed" if off else "")
    params = {"bin": f"{float(data['bin_ms']):g} ms",
              "shots/position": int(data["nshot"]),
              "T_e assumed": f"{float(data['te_ev']):g} eV",
              "multi-electrode bias": f"{bias[0]:g}-{bias[1]:g} ms"}
    if off:
        params["offset removed"] = (
            f"({off[0]:+.3f}, {off[1]:+.3f}) km/s, the plane mean over "
            f"{PREBIAS_MACHINE_MS[0]:g}-{PREBIAS_MACHINE_MS[1]:g} ms machine")
    return write_flow_slider(
        out or slider_path(name),
        pos_x=data["pos_x"], pos_y=data["pos_y"], vx=vx, vy=vy,
        v_r=v_r, v_th=v_th,
        t_axis=machine_ms(data["t_ms"]), axis_label="time (from discharge)",
        title=f"Run {run} - calibrated Mach flow, P29 plane"
              + (" (pre-bias offset removed)" if off else ""),
        source=str(data["source_file"]),
        params=params,
        quiver_step=quiver_step, vmax=vmax, extra_fields=extra)


def slider_path(name):
    """Centralized ``.html`` location, beside this campaign's PNGs.

    A slider page is rendered from the npz processed data.
    """
    return fig_path(name, FIG_SUBDIR, ext=".html")


def plot_flow_frame(t_ms, npz_path=None, quiver_step=QUIVER_STEP, vmax=None,
                    machine_time=True, save_fig=True):
    """Static figure at the bin nearest ``t_ms``: in-plane, v_theta, v_r.

    Same panels in the same order as :func:`emit_flow_slider`, so a figure cannot
    show something different from the page it was scrubbed on.
    """
    data = load_flow(npz_path)
    want = t_ms - float(data["daq_trigger_ms"]) if machine_time else t_ms
    k = int(np.argmin(np.abs(data["t_ms"] - want)))
    cx, cy, _ratio = find_flow_centre(data)
    v_r, v_th = polar_components(data["v_X"][:, k], data["v_Y"][:, k],
                                 data["pos_x"], data["pos_y"], (cx, cy))

    bias = np.asarray(data["bias_machine_ms"], float)
    t_mach = machine_ms(data["t_ms"][k])
    on = "bias on" if bias[0] <= t_mach <= bias[1] else "bias off"
    run_num = run_num_of(str(data["source_file"]))
    # A frame past the signal floor is labelled as such on the figure itself:
    # the map still looks like a flow field, so the caveat has to travel with it.
    if not bool(data["reliable"][k]):
        suptitle = (f"Run {run_num}  -  t = {t_mach:.2f} ms from discharge  -  "
                    f"NOT A FLOW MEASUREMENT: summed tip current is below "
                    f"{float(data['min_signal_frac']):.0%} of peak here; this "
                    f"is the ratio of two decayed traces")
        color = "firebrick"
    else:
        suptitle = (f"Run {run_num}  -  t = {t_mach:.2f} ms from discharge "
                    f"({on}; bias {bias[0]:g}-{bias[1]:g} ms)  -  "
                    f"{_te_note(data)}  -  polar centre "
                    f"({cx:+.2f}, {cy:+.2f}) cm")
        color = None

    fig = _draw_flow_frame(data["pos_x"], data["pos_y"], data["v_X"][:, k],
                           data["v_Y"][:, k], v_r, v_th, (cx, cy), suptitle,
                           quiver_step=quiver_step, vmax=vmax,
                           suptitle_color=color)
    # 'p' for the decimal point: a '.' in the stem is read as a file extension.
    name = f"{run_num}-mach-flow-{t_mach:.2f}ms".replace(".", "p")
    finalize_figure(fig, save_fig=resolve_save(save_fig, name, FIG_SUBDIR))


if __name__ == "__main__":
    if not os.path.exists(flow_npz_path()):
        batch_flow()
    print(emit_flow_slider())
    print(emit_flow_slider(subtract_offset=True))
