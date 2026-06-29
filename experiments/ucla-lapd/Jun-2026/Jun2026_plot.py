"""Jun-2026 LAPD plotting -- shared figure module.

The **plotting** home for the Jun-2026 LAPD analyses.  The processing modules
(e.g. ``Jun2026_IV.py``) read raw HDF5, do the heavy lifting, and save results
to ``.npz``; this module reads those saved arrays back and draws the figures.
It does **no** processing of its own.

TODO: Show plot not working

Layout
------
* **Shared helpers** (top): figure finalisation (:func:`finalize_figure`),
  centralized output paths (:func:`fig_path`), and run titling from the
  hand-written description (:func:`run_title`).  Reuse these in every plot so the
  save/show behaviour and figure locations stay consistent.
* **Per-figure sections** below: one section per figure type, each with its
  ``plot_*`` drawing function and a ``plot_*_run`` driver that loads the saved
  ``.npz`` and calls it.  The IV line-scan is the first such section; add new
  Jun-2026 figures as further sections following the same shape.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt

from data_analysis.io import open_lapd
from data_analysis.io.paths import output_path
from data_analysis.signal.core import downsample_blockmean

import Jun2026_IV as jiv

# Default subdirectory under $DATA_ANALYSIS_OUTPUT/figures/ for Jun-2026 figures.
FIG_SUBDIR = "Jun2026"

# Isat reference panel (4th panel of the IV line-scan figure).  A fixed-bias
# ion-saturation tip on a *stationary* probe -- every position is essentially
# identical -- so we read one representative position (npos // 2) and draw a
# SINGLE shot's trace, block-mean downsampled, over 0..ISAT_TMAX_MS ms with the
# IV sweep result times scattered on it.  The channel is ~2.5M samples/shot, so
# reading every shot / plotting the raw trace is what made plotting hang; one
# block-mean-downsampled shot is fast and enough for a visual reference.  No
# shot averaging and no current scaling.  Channel is hardcoded here for now.
# These are plotting config (which reference trace to show), so they live with
# the plotting code.
ISAT_SCOPE = "machscope"   # scope group holding the fixed-bias Isat channel
ISAT_CHAN = "C2"           # Isat channel name
ISAT_SHOT = 0              # which single shot to show (0-based, within the position)
ISAT_DS_Q = 500            # block-mean downsample factor (2.5M -> ~5k samples)
ISAT_TMAX_MS = 10.0        # x-axis upper limit for the Isat panel, ms


# =========================================================================== #
#  Shared helpers -- reuse these in every figure section below.
# =========================================================================== #

def finalize_figure(fig, save_fig=None):
    """Save and/or show a finished figure, then release it.

    ``save_fig`` -- path to write the PNG to; ``None`` skips saving.
    ``show``     -- call ``plt.show()`` (set False for headless/batch saving).
    """
    fig.tight_layout()
    if save_fig is not None:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_fig}")
    plt.close(fig)


def fig_path(name, subdir=FIG_SUBDIR):
    """Centralized figure location under the repo-external output root.

    Routes through :func:`data_analysis.io.paths.output_path`, so figures land in
    ``$DATA_ANALYSIS_OUTPUT/figures/<subdir>/`` (default ``~/data-analysis-output``)
    rather than next to the raw data or in the repo.  ``name`` is the filename
    stem already including any run/tip/figure tags (e.g. ``"02-tipR-line"``); a
    ``.png`` extension is added if absent.

    Drivers follow the convention: ``save_fig is True`` -> ``fig_path(name)``; a
    string -> that explicit path; anything falsey -> don't save.
    """
    if not os.path.splitext(name)[1]:
        name += ".png"
    return output_path("figures", subdir, name)


# The gas-puff setting the operator writes into the run ``description`` -- e.g.
# "Helium backside pressure 40 Psi, Puff voltage 75V for 25ms West+East".  We
# title each plot with the "Puff voltage ... ms" span of that line; it is the
# knob that varies run to run.
_PUFF_RE = re.compile(r"puff voltage.*?\d+\s*ms", re.IGNORECASE)


def run_title(ifn, run_num, run=None):
    """Plot title for a run: ``"<run_num>-<puff voltage ... ms>"``.

    Reads this run's own hand-written ``description`` and pulls the gas-puff span
    (``"Puff voltage 75V for 25ms"``) -- the setting that changes run to run -- so
    e.g. run 01 titles as ``"01-Puff voltage 75V for 25ms"``.  Returns just the
    run number when the description has no recognizable puff line, and ``None``
    when the file can't be read / isn't pydaq, so the plot falls back to its
    generic title rather than being blocked by a description hiccup.

    Pass an already-open ``run`` to reuse it instead of re-opening ``ifn`` (the
    HDF5 is multi-GB); ``None`` opens it here.  Generic across Jun-2026 figures.
    """
    try:
        if run is None:
            run = open_lapd(ifn)
        desc = run.description()
    except (OSError, ValueError, NotImplementedError, KeyError) as e:
        print(f"  (title: could not read run description -- {e})")
        return None
    for sec in desc.sections.values():
        for it in sec.items:
            m = _PUFF_RE.search(it.raw)
            if m:
                return f"{run_num}-{m.group(0)}"
    return run_num


def run_num_of(ifn):
    """The leading run-number token of a run file (``"02-He-..."`` -> ``"02"``)."""
    return os.path.basename(ifn).split("-")[0]


def discover_tips(ifn):
    """The tip labels :func:`Jun2026_IV.process_run` produced ``.npz`` for.

    Mirrors that processing's tip selection so a driver plots exactly the tips
    that were saved: the override tip when ``I_CHAN``/``V_CHAN`` are set, else
    every complete I+V pair discovered in the file.
    """
    if jiv.I_CHAN is not None and jiv.V_CHAN is not None:
        return ["override"]
    scope_name = jiv.SCOPE_NAME if jiv.SCOPE_NAME is not None else jiv.find_lp_scope(ifn)
    pairs, _ = jiv.discover_lp_channels(ifn, scope_name)
    return list(pairs)


# =========================================================================== #
#  Figure: IV line-scan (Vp / Te / ne vs x + Isat reference)
# =========================================================================== #

def plot_iv_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, tndx_list, save_fig=None,
                 show=True, title=None, isat=None):
    """
    Plot the line scan: Vp, Te, and ne vs x at selected sweep (time) indices,
    plus a reference Isat trace.

    Jun-2026 LP runs are a 1-D line scan (y == 0), so each row of the parameter
    arrays already corresponds to one x position -- no center-line extraction is
    needed (unlike Mar-2026's xy-plane).  ``Vp_arr`` etc. are (n_locs, n_sweeps)
    with ``n_locs == len(xpos)``.

    A 4th panel shows the fixed-bias Isat trace (a single block-mean-downsampled
    shot at the stationary probe's mid position, from :func:`_read_isat_trace`)
    over 0..``ISAT_TMAX_MS`` ms, with the IV sweep result times
    (``t_ls[tndx_list]``) scattered *on the line* as points coloured to match the
    IV panels.  ``isat`` is the downsampled ``(tarr, I)`` in (seconds, current);
    pass ``None`` to leave that panel blank (e.g. when the Isat channel is
    absent).

    ``save_fig`` -- a path to write the figure to (PNG); ``None`` skips saving.
    ``show`` -- call ``plt.show()`` (set False for headless/batch saving only).
    ``title`` -- figure title; the caller passes the run's gas-puff label (run
    number + the puff setting, e.g. ``"01-Puff voltage 75V for 25ms"``).  ``None``
    uses a generic default.
    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 13))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(tndx_list)))

    # Track the (colour, time-in-ms) of each IV result time so the Isat panel can
    # mark the same instants the Vp/Te/ne panels show.
    marked_times = []
    for color, t_idx in zip(colors, tndx_list):
        if t_idx >= Vp_arr.shape[1]:
            print(f"Warning: time index {t_idx} exceeds array size {Vp_arr.shape[1]}")
            continue

        t_val = t_ls[t_idx] * 1e3  # Convert to ms
        label = f"t = {t_val:.2f} ms"
        marked_times.append((color, t_val))

        axs[0].plot(xpos, Vp_arr[:, t_idx], "o-", color=color, label=label,
                    linewidth=2, markersize=5)
        axs[1].plot(xpos, Te_arr[:, t_idx], "s-", color=color, label=label,
                    linewidth=2, markersize=5)
        axs[2].plot(xpos, ne_arr[:, t_idx], "^-", color=color, label=label,
                    linewidth=2, markersize=5)

    # Title is the experiment-difference string from the caller (run number +
    # the one changed setting vs the baseline, e.g. "01: puff voltage 75V for
    # 25ms"); fall back to a generic label when no difference was resolved.
    base_title = title or "Plasma Parameters along LP line scan (y = 0)"
    axs[0].set_title(base_title, fontsize=12, fontweight="bold")
    axs[0].set_ylabel("Vp [V]", fontsize=11)
    axs[1].set_ylabel("Te [eV]", fontsize=11)
    axs[2].set_ylabel("ne [cm$^{-3}$]", fontsize=11)
    axs[2].set_xlabel("X Position [cm]", fontsize=11)
    for ax in axs[:3]:
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.sharex(axs[0])

    _draw_isat_panel(axs[3], isat, marked_times)

    finalize_figure(fig, save_fig=save_fig)


def _draw_isat_panel(ax, isat, marked_times):
    """Draw the Isat reference panel (4th panel of the IV line-scan figure).

    ``isat`` is the downsampled ``(tarr, I)`` in (seconds, current) from
    :func:`_read_isat_trace`, or ``None`` to leave the panel blank with a note.
    ``marked_times`` is the list of ``(colour, time_ms)`` of the IV result
    times; each is drawn as a scatter point *on the Isat line* (its current
    value interpolated at that time) so the IV sweep instants are visible on the
    reference trace.  The panel is clipped to 0..``ISAT_TMAX_MS`` ms.
    """
    if isat is not None:
        t_ms = np.asarray(isat[0]) * 1e3
        I = np.asarray(isat[1])
        ax.plot(t_ms, I, "k", linewidth=1.0, zorder=1)
        # Scatter each IV result time onto the line (interpolate its Isat value).
        for color, t_val in marked_times:
            if t_ms[0] <= t_val <= t_ms[-1]:
                I_at = np.interp(t_val, t_ms, I)
                ax.scatter(t_val, I_at, color=color, s=80, zorder=3,
                           edgecolors="k", linewidths=0.5)
        ax.set_title(f"Isat reference (stationary probe, "
                     f"'{ISAT_SCOPE}'/{ISAT_CHAN}, single shot @ mid position)",
                     fontsize=10)
    else:
        ax.text(0.5, 0.5,
                f"Isat channel '{ISAT_SCOPE}'/{ISAT_CHAN} unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="0.5")
    ax.set_xlim(0, ISAT_TMAX_MS)
    ax.set_ylabel("Isat [a.u.]", fontsize=11)
    ax.set_xlabel("t [ms]", fontsize=11)
    ax.grid(True, alpha=0.3)


def _read_isat_trace(run, npos, nshot):
    """Read + downsample a SINGLE-shot Isat reference trace for the 4th panel.

    The whole point is to be fast and never hang.  Steps:

    1. Select the hardcoded fixed-bias Isat channel (``ISAT_SCOPE`` /
       ``ISAT_CHAN``) at one representative position (``npos // 2`` -- the probe
       is stationary, so every position is essentially identical).
    2. Load just **one** shot (``ISAT_SHOT``) off disk -- ~2.5M samples.
    3. Block-mean downsample the current array by ``ISAT_DS_Q`` (average each
       block of Q samples -- anti-aliases as it shrinks the trace to ~5k pts).
    4. Stride downsample the time array by the same Q (the time axis is uniform,
       so plain striding lines it up with the block-mean current).

    No shot averaging and no current scaling.  Takes the already-open ``run`` (so
    the HDF5 isn't re-opened just for this).  Returns ``(tarr, I)`` -- the
    downsampled time (seconds) and current -- or ``None`` if the channel/scope
    isn't present (so the caller draws the panel blank).
    """
    pos_index = npos // 2
    shot = pos_index * nshot + ISAT_SHOT
    try:
        Istack, tarr = run.channel(ISAT_CHAN, scope_name=ISAT_SCOPE,
                                   shots=slice(shot, shot + 1))
    except (OSError, ValueError, KeyError) as e:
        print(f"  (Isat panel: channel '{ISAT_SCOPE}'/{ISAT_CHAN} unavailable -- {e})")
        return None
    I = Istack[0]
    # block-mean downsample the current (anti-aliases as it shrinks the trace);
    # the time axis rides along on the same blocks.
    t_ds, I_ds = downsample_blockmean(tarr, I, ISAT_DS_Q)
    return t_ds, I_ds


def _plot_iv_line_tip(data_dir, run_num, tip, ifn=None, tndx_list=None,
                      save_fig=True, show=True):
    """Load one tip's saved IV arrays and draw the line-scan plot (no reprocessing).

    ``tip`` is the discovered tip label, or ``"override"``/``None`` for the
    un-tagged single-tip files.  ``ifn`` (the run's HDF5 path) drives the
    descriptive title (just the changed setting vs the baseline) via
    :func:`run_title` and the Isat reference read; pass ``None`` for the plot's
    generic title and a blank Isat panel.  ``save_fig`` True routes the PNG
    through :func:`fig_path`; pass a path to override, or False to skip saving.
    """
    load_tip = None if tip in (None, "override") else tip
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = jiv.load_data(data_dir, run_num, tip=load_tip)
    _, _, _, xpos, _, npos, nshot = jiv.load_sweep_data(data_dir, run_num, tip=load_tip)

    ndx = tndx_list if tndx_list is not None else list(
        range(0, Vp_arr.shape[1], max(1, Vp_arr.shape[1] // 4)))

    # Open the run once (multi-GB HDF5) and reuse it for both raw reads below:
    # the descriptive title and the Isat reference trace.  None -> no run file,
    # so a generic title and a blank Isat panel.
    run = open_lapd(ifn) if ifn is not None else None

    # Title = "<run_num>-<puff voltage ... ms>" from this run's description (e.g.
    # "01-Puff voltage 75V for 25ms"); no tip.  None -> the plot's generic
    # default.  (Output filenames still carry run_num + tip, via the name below.)
    title = run_title(ifn, run_num, run=run) if run is not None else None

    # Reference Isat trace for the 4th panel -- read from the raw HDF5 (it's not
    # in the saved .npz).  None when there's no run file or the channel is absent,
    # in which case the panel is drawn blank.
    isat = _read_isat_trace(run, npos, nshot) if run is not None else None

    name = f"{run_num}{jiv._tip_tag(load_tip)}-line"
    save_path = fig_path(name) if save_fig is True else (save_fig or None)
    plot_iv_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, ndx,
                 save_fig=save_path, title=title, isat=isat)


def plot_iv_line_run(ifn, tndx_list=None, save_fig=True, show=True):
    """Draw the IV line-scan plot(s) for a run from already-saved ``.npz``.

    Plots each tip :func:`Jun2026_IV.process_run` saved (see
    :func:`discover_tips`) from its ``-tip<T>-*.npz`` files -- no HDF5 read, no
    reprocessing (apart from the small Isat reference trace).  Run
    :func:`Jun2026_IV.process_run` first to create the ``.npz``.  ``save_fig``
    True writes the PNG via :func:`fig_path`; pass a path to override or False to
    skip.
    """
    data_dir = os.path.dirname(ifn)
    run_num = run_num_of(ifn)

    for tip in discover_tips(ifn):
        print(f"Plotting IV line-scan for tip {tip} from saved data...")
        _plot_iv_line_tip(data_dir, run_num, tip, ifn=ifn, tndx_list=tndx_list,
                          save_fig=save_fig)


# =========================================================================== #
#  Add further Jun-2026 figure sections below, each following the IV pattern:
#    plot_<fig>(...)            -- draw one figure, ending in finalize_figure()
#    plot_<fig>_run(ifn, ...)   -- load the saved .npz and call plot_<fig>
# =========================================================================== #


if __name__ == '__main__':

    ifn = r"D:\data\LAPD\jun2026-jia\05-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"

    if not ifn:
        raise SystemExit("Set `ifn` to a run file whose .npz has been produced "
                         "by Jun2026_IV.process_run first.")

    # Plot from the saved .npz (run Jun2026_IV.process_run first to create them).
    # save_fig=True -> write the PNG via fig_path; save_fig=False -> just show.
    plot_iv_line_run(ifn, tndx_list=[-7, -5, -3, -1], save_fig=True)
