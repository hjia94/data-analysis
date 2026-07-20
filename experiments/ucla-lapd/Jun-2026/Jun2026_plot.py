"""Jun-2026 LAPD plotting -- shared figure module.

The **plotting** home for the Jun-2026 LAPD analyses.  The processing modules
(e.g. ``Jun2026_IV.py``) read raw HDF5, do the heavy lifting, and save results
to ``.npz``; this module reads those saved arrays back and draws the figures.
It does **no** processing of its own.

Figure functions (each reads the relevant run's saved ``.npz`` -- process first):

* ``plot_iv_line``      -- one IV line scan: Vp, Te, ne (+ Te*ne) vs x at a few
  selected sweep times, with a reference Isat trace.  The low-level drawer that
  takes arrays directly.
* ``plot_iv_line_run``  -- ``plot_iv_line`` for a whole run: draws one line-scan
  figure per probe tip from that run's saved IV npz (calibrated ne by default).
* ``plot_iv_isat_combined`` -- 6-panel combined figure: the IV line scan
  (panels 1-4, Vp/Te/ne/Te*ne, calibrated ne by default) plus the Isat trace and
  its FFT (panels 5-6) from a separate Isat run.
* ``plot_xcorr_plane_run`` -- two-channel cross-spectrum over an xy-plane run:
  broadband coherence and cross-phase as side-by-side imshow maps (Smith-1974).
* ``plot_xcorr_band_plane_run`` -- the narrow-band version: band-averaged
  coherence and phase-difference (+ tracked peak frequency) xy-plane maps.
"""

import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from data_analysis.io import open_lapd, parse_gas_puff
from data_analysis.plasma.langmuir import (
    load_plasma_data, load_ne_calibrated, load_sweep_axes, tip_tag)
from data_analysis.signal.core import downsample_blockmean
from data_analysis.utils import run_num_of
from data_analysis.viz import plot_utils
from data_analysis.viz.plot_utils import finalize_figure, grid_by_position

import Jun2026_IV as jiv
import Jun2026_Isat as jis
import Jun2026_xcorr as jxc

# Default subdirectory under $DATA_ANALYSIS_OUTPUT/figures/ for Jun-2026 figures.
FIG_SUBDIR = "Jun2026"

# Isat reference panel (5th panel of the IV line-scan figure).  A fixed-bias
# ion-saturation tip on a *stationary* probe -- every position is essentially
# identical -- so we read one representative position (npos // 2) and draw a
# SINGLE shot's trace, block-mean downsampled, over 0..ISAT_TMAX_MS ms with the
# IV sweep result times scattered on it.  The channel is ~2.5M samples/shot, so
# reading every shot / plotting the raw trace is what made plotting hang; one
# block-mean-downsampled shot is fast and enough for a visual reference.  No
# shot averaging and no current scaling.  These are plotting config (which
# reference trace to show), so they live with the plotting code -- but the
# channel identity itself is owned by Jun2026_Isat (the processing module), so
# changing it there moves both the batch FFT and these reference panels.
ISAT_SCOPE = jis.SCOPE_NAME   # scope group holding the fixed-bias Isat channel
ISAT_CHAN = jis.CHAN          # Isat channel name
ISAT_SHOT = 0              # which single shot to show (0-based, within the position)
ISAT_DS_Q = 500            # block-mean downsample factor (2.5M -> ~5k samples)
ISAT_TMAX_MS = 10.0        # x-axis upper limit for the Isat panel, ms


# =========================================================================== #
#  Shared helpers -- thin bindings of the data_analysis.viz helpers to this
#  campaign's FIG_SUBDIR; reuse these in every figure section below.
# =========================================================================== #

def fig_path(name, subdir=FIG_SUBDIR):
    """Centralized figure location: :func:`data_analysis.viz.plot_utils.fig_path`
    with this campaign's default subdirectory."""
    return plot_utils.fig_path(name, subdir)


def _resolve_save(save_fig, name):
    """Resolve a driver's ``save_fig`` convention (True -> :func:`fig_path`, a
    string -> that path, falsey -> None).  Every ``plot_*_run`` driver resolves
    through here so the convention lives in one place."""
    return plot_utils.resolve_save(save_fig, name, FIG_SUBDIR)


def _run_description(ifn, run=None):
    """A run's parsed ``description``, or ``None`` (with a note) if unreadable.

    Pass an already-open ``run`` to reuse it instead of re-opening ``ifn``;
    ``None`` opens it here.  Shared guard for the title helpers, so a
    description hiccup degrades to a generic title instead of blocking the plot.
    """
    try:
        if run is None:
            run = open_lapd(ifn)
        return run.description()
    except (OSError, ValueError, NotImplementedError, KeyError) as e:
        print(f"  (title: could not read run description -- {e})")
        return None


def _puff_label(puff):
    """Format a ``(puff_v, puff_t)`` pair as the operator writes it."""
    return f"Puff voltage {puff[0]:g}V for {puff[1]:g}ms"


def run_title(ifn, run_num, run=None):
    """Plot title for a run: ``"<run_num>-<puff voltage ... ms>"``.

    Reads this run's own hand-written ``description`` and pulls the gas-puff
    setting (via the shared :func:`data_analysis.io.parse_gas_puff` -- the same
    parser ``group_by_gas_puff`` uses) -- the knob that changes run to run -- so
    e.g. run 01 titles as ``"01-Puff voltage 75V for 25ms"``.  Returns just the
    run number when the description has no recognizable puff line, and ``None``
    when the file can't be read / isn't pydaq, so the plot falls back to its
    generic title rather than being blocked by a description hiccup.
    """
    desc = _run_description(ifn, run=run)
    if desc is None:
        return None
    puff = parse_gas_puff(desc.raw)
    return f"{run_num}-{_puff_label(puff)}" if puff else run_num


def discover_tips(ifn):
    """The tip labels :func:`Jun2026_IV.process_run` produced ``.npz`` for.

    Read off the saved ``<run>-tip<T>-sweep-data.npz`` filenames rather than by
    re-running channel discovery against the raw multi-GB HDF5 -- the saved
    files *are* what can be plotted (a tip whose processing failed or was
    skipped has no npz).  An un-tagged ``<run>-sweep-data.npz`` (the override /
    legacy single-tip case) maps to ``["override"]``.
    """
    data_dir, run_num = os.path.dirname(ifn), run_num_of(ifn)
    tip_re = re.compile(rf"{re.escape(run_num)}-tip(.+)-sweep-data\.npz$")
    tips = sorted(m.group(1) for m in
                  (tip_re.match(os.path.basename(p)) for p in
                   glob.glob(os.path.join(data_dir, f"{run_num}-tip*-sweep-data.npz")))
                  if m)
    if not tips:
        if os.path.isfile(os.path.join(data_dir, f"{run_num}-sweep-data.npz")):
            return ["override"]
        raise FileNotFoundError(
            f"no saved sweep npz for run {run_num} in {data_dir}; run "
            "Jun2026_IV.process_run(ifn) first")
    return tips


# =========================================================================== #
#  Figure: IV line-scan (Vp / Te / ne vs x + Isat reference)
# =========================================================================== #

def _draw_iv_panels(axs4, Vp_arr, Te_arr, ne_arr, xpos, t_ls, tndx_list, title):
    """Draw the Vp / Te / ne / Te*ne vs x panels at the selected sweep times.

    ``axs4`` is the four axes to draw into (Vp, Te, ne, Te*ne, top to bottom).
    The fourth panel is the pressure-like product Te*ne derived from the Te and
    ne arrays (panels 2 and 3).  Each ``tndx_list`` time index is plotted in a
    rainbow colour across all four panels; the title goes on the top panel and
    the x-label on the bottom one.  Returns ``marked_times`` -- the
    ``(colour, time_ms)`` of each drawn time, so a caller (the IV line-scan's
    Isat panel) can mark the same instants elsewhere.  Shared by
    :func:`plot_iv_line` and :func:`plot_iv_isat_combined` so the line-scan
    styling lives in one place.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tndx_list)))
    Tene_arr = Te_arr * ne_arr   # pressure-like product, derived from panels 2-3
    panels = [(axs4[0], Vp_arr, "Vp [V]", "o-"),
              (axs4[1], Te_arr, "Te [eV]", "s-"),
              (axs4[2], ne_arr, "ne [cm$^{-3}$]", "^-"),
              (axs4[3], Tene_arr, "Te*ne [eV cm$^{-3}$]", "D-")]

    marked_times = []
    for color, t_idx in zip(colors, tndx_list):
        if t_idx >= Vp_arr.shape[1]:
            print(f"Warning: time index {t_idx} exceeds array size {Vp_arr.shape[1]}")
            continue
        t_val = t_ls[t_idx] * 1e3  # Convert to ms
        label = f"t = {t_val:.2f} ms"
        marked_times.append((color, t_val))
        for ax, arr, _, marker in panels:
            ax.plot(xpos, arr[:, t_idx], marker, color=color, label=label,
                    linewidth=2, markersize=5)

    axs4[0].set_title(title or "Plasma Parameters along LP line scan (y = 0)",
                      fontsize=12, fontweight="bold", pad=30)
    for ax, _, ylabel, _ in panels:
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.sharex(axs4[0])
    # Panels share one x-axis: hide the redundant tick labels on all but the
    # bottom panel so only it carries the X-position ticks and label.
    for ax in axs4[:-1]:
        ax.tick_params(labelbottom=False)
    axs4[-1].set_xlabel("X Position [cm]", fontsize=11)
    # (Panels 1-4 share the x-axis, so the gaps between them are dead space; the
    # caller passes these axes to finalize_figure(compact_axes=...) to pack them
    # tight *after* tight_layout, leaving any trailing Isat/FFT panels alone.)
    # Panels share identical time labels, so draw one legend (taken from the top
    # panel's handles) horizontally above panel 1, tucked between it and the
    # figure title (the title ``pad`` above leaves room so they don't overlap).
    handles, labels = axs4[0].get_legend_handles_labels()
    axs4[0].legend(handles, labels, fontsize=9, loc="lower center",
                   bbox_to_anchor=(0.5, 1.01), ncol=len(labels))
    return marked_times


def plot_iv_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, tndx_list, save_fig=None,
                 show=False, title=None, isat=None, fft=None,
                 fft_fmax_khz=80.0):
    """
    Plot the line scan: Vp, Te, and ne vs x at selected sweep (time) indices,
    plus a reference Isat trace.

    Jun-2026 LP runs are a 1-D line scan (y == 0), so each row of the parameter
    arrays already corresponds to one x position -- no center-line extraction is
    needed (unlike Mar-2026's xy-plane).  ``Vp_arr`` etc. are (n_locs, n_sweeps)
    with ``n_locs == len(xpos)``.

    The 5th panel shows the fixed-bias Isat trace (a single block-mean-downsampled
    shot at the stationary probe's mid position, from :func:`_read_isat_trace`)
    over 0..``ISAT_TMAX_MS`` ms, with the IV sweep result times
    (``t_ls[tndx_list]``) scattered *on the line* as points coloured to match the
    IV panels.  ``isat`` is the downsampled ``(tarr, I)`` in (seconds, current);
    pass ``None`` to leave that panel blank (e.g. when the Isat channel is
    absent).

    The last panel shows that same Isat channel's all-shot-averaged FFT (like
    :func:`plot_iv_isat_combined`), clipped to ``0..fft_fmax_khz`` kHz.  ``fft`` is
    the ``(freq_khz, amp)`` spectrum; pass ``None`` to leave it blank (e.g. when
    the Isat channel is absent).

    ``save_fig`` -- a path to write the figure to (PNG); ``None`` skips saving.
    ``show`` -- call ``plt.show()`` (default False: headless/batch saving).
    ``title`` -- figure title; the caller passes the run's gas-puff label (run
    number + the puff setting, e.g. ``"01-Puff voltage 75V for 25ms"``).  ``None``
    uses a generic default.
    """
    fig, axs = plt.subplots(6, 1, figsize=(12, 18))

    # Title is the experiment-difference string from the caller (run number + the
    # one changed setting vs the baseline, e.g. "01: puff voltage 75V for 25ms").
    # The Isat panel marks the same instants the Vp/Te/ne/Te*ne panels show.
    marked_times = _draw_iv_panels(axs[:4], Vp_arr, Te_arr, ne_arr, xpos, t_ls,
                                   tndx_list, title)
    _draw_isat_panel(axs[4], isat, marked_times,
                     title=f"Isat reference (stationary probe, "
                           f"'{ISAT_SCOPE}'/{ISAT_CHAN}, single shot @ mid position)")
    _draw_fft_panel(axs[5], fft, fft_fmax_khz,
                    f"Isat averaged FFT (all shots), '{ISAT_SCOPE}'/{ISAT_CHAN}")

    finalize_figure(fig, save_fig=save_fig, show=show, compact_axes=axs[:4])


def _scatter_iv_times_on_isat(ax, t_ms, I, marked_times):
    """Scatter each IV result time as a point *on* an Isat trace.

    ``t_ms`` / ``I`` are the Isat trace (time in ms, current).  For each
    ``(colour, time_ms)`` in ``marked_times`` (from :func:`_draw_iv_panels`),
    drops a point at that time on the line -- its current interpolated from the
    trace -- coloured to match the IV panels, so the IV sweep instants are
    visible on the Isat curve.  Shared by the line-scan and combined figures.
    """
    for color, t_val in marked_times:
        if t_ms[0] <= t_val <= t_ms[-1]:
            I_at = np.interp(t_val, t_ms, I)
            ax.scatter(t_val, I_at, color=color, s=80, zorder=3,
                       edgecolors="k", linewidths=0.5)


def _draw_isat_panel(ax, isat, marked_times, title=None, tmax_ms=ISAT_TMAX_MS,
                     ylim=None):
    """Draw an Isat reference panel (5th panel of the line-scan / combined figures).

    ``isat`` is the ``(tarr, I)`` trace in (seconds, current) -- downsampled
    single shot or shot-averaged, per the caller -- or ``None`` to leave the
    panel blank with a note.  ``marked_times`` is the list of
    ``(colour, time_ms)`` of the IV result times; each is drawn as a scatter
    point *on the Isat line* (its current value interpolated at that time) so
    the IV sweep instants are visible on the reference trace.  The panel is
    clipped to 0..``tmax_ms``; ``title``/``ylim`` are optional.
    """
    if isat is not None:
        t_ms = np.asarray(isat[0]) * 1e3
        I = np.asarray(isat[1])
        ax.plot(t_ms, I, "k", linewidth=1.0, zorder=1)
        # Scatter each IV result time onto the line (interpolate its Isat value).
        _scatter_iv_times_on_isat(ax, t_ms, I, marked_times)
        if title:
            ax.set_title(title, fontsize=10)
    else:
        ax.text(0.5, 0.5,
                f"Isat channel '{ISAT_SCOPE}'/{ISAT_CHAN} unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="0.5")
    ax.set_xlim(0, tmax_ms)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("Isat [a.u.]", fontsize=11)
    ax.set_xlabel("t [ms]", fontsize=11)
    ax.grid(True, alpha=0.3)


def _draw_fft_panel(ax, fft, fmax_khz, title, chan_label=None):
    """Draw an Isat-FFT panel (last panel of the IV line-scan / combined figures).

    ``fft`` is the ``(freq_khz, amp)`` shot-averaged amplitude spectrum (frequency
    in kHz), or ``None`` to leave the panel blank with a note.  The y-axis is log
    (``semilogy``) and the x-axis is clipped to ``0..fmax_khz``.  Shared by
    :func:`plot_iv_line` and :func:`plot_iv_isat_combined` so the FFT panel looks
    the same in both figures.  ``chan_label`` names the channel in the blank-panel
    note; pass the caller's own scope/chan (the combined figure can differ from
    the module defaults), or ``None`` for ``'ISAT_SCOPE'/ISAT_CHAN``.
    """
    if fft is not None:
        freq_khz, amp = fft
        ax.semilogy(freq_khz, amp, "k", linewidth=0.9)
        ax.set_title(title, fontsize=10)
    else:
        ax.text(0.5, 0.5,
                f"Isat channel {chan_label or f'{ISAT_SCOPE!r}/{ISAT_CHAN}'} unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="0.5")
    ax.set_xlim(0, fmax_khz)
    ax.set_ylabel("Isat amplitude [a.u.]", fontsize=11)
    ax.set_xlabel("frequency [kHz]", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)


def _read_isat_trace(run, npos, nshot):
    """Read + downsample a SINGLE-shot Isat reference trace for the 5th panel.

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
    if Istack is None:   # run.channel signals a missing scope with (None, None)
        print(f"  (Isat panel: channel '{ISAT_SCOPE}'/{ISAT_CHAN} unavailable)")
        return None
    I = Istack[0]
    # block-mean downsample the current (anti-aliases as it shrinks the trace);
    # the time axis rides along on the same blocks.
    t_ds, I_ds = downsample_blockmean(tarr, I, ISAT_DS_Q)
    return t_ds, I_ds


def _read_isat_fft(ifn, fft_npz=None):
    """Load this run's all-shot-averaged Isat FFT for the figure's last panel.

    Reads the precomputed spectrum from the batch FFT npz
    (:func:`Jun2026_Isat.batch_fft`, default ``<run dir>/Jun2026_Isat.OUT_NPZ``)
    rather than recomputing it -- the run's own ``<basename>__amp`` entry against
    the shared ``freq`` axis, same as :func:`plot_iv_isat_combined`.  Returns
    ``(freq_khz, amp)`` -- amplitude vs frequency in kHz -- or ``None`` if the npz
    or this run's entry is missing (so :func:`_draw_fft_panel` draws it blank).
    """
    if fft_npz is None:
        fft_npz = os.path.join(os.path.dirname(ifn), jis.OUT_NPZ)
    run_key = os.path.splitext(os.path.basename(ifn))[0]
    try:
        with np.load(fft_npz) as fft:   # NpzFile holds an open fd; close it
            return fft["freq"] * 1e-3, fft[f"{run_key}__amp"]
    except (OSError, KeyError) as e:
        print(f"  (FFT panel: no saved FFT for '{run_key}' in {fft_npz} -- {e})")
        return None


def _load_ne(data_dir, run_num, load_tip, raw_ne, calibrated):
    """Pick the ne to plot: interferometer-calibrated if available, else raw.

    ``calibrated`` True swaps in ``ne_cal_arr`` (written by
    :func:`data_analysis.plasma.langmuir.calibrate_plasma_npz`).  If the run
    hasn't been calibrated yet the plasma npz has no calibrated array, so we
    fall back to ``raw_ne`` (the array from :func:`load_plasma_data`) and print
    a note pointing at the calibration step.  ``calibrated`` False just returns
    ``raw_ne``.
    """
    if not calibrated:
        return raw_ne
    try:
        return load_ne_calibrated(data_dir, run_num, tip=load_tip)[0]
    except KeyError:
        print(f"  (ne: run {run_num}{tip_tag(load_tip)} not calibrated yet -- "
              "plotting raw ne; run Jun2026_IV.calibrate_plasma_npz to calibrate)")
        return raw_ne


def _plot_iv_line_tip(data_dir, run_num, tip, tndx_list=None, save_fig=True,
                      show=False, title=None, isat=None, fft=None,
                      calibrated=True):
    """Load one tip's saved IV arrays and draw the line-scan plot (no reprocessing).

    ``tip`` is the discovered tip label, or ``"override"``/``None`` for the
    un-tagged single-tip files.  ``title`` / ``isat`` / ``fft`` are the
    tip-invariant pieces the run driver reads once (descriptive title, Isat
    reference trace, saved FFT); each may be ``None`` for a generic title /
    blank panel.  ``save_fig`` True routes the PNG through :func:`fig_path`;
    pass a path to override, or False to skip saving.  ``calibrated`` True
    (default) plots the interferometer-calibrated ne when it exists, falling
    back to raw ne otherwise (see :func:`_load_ne`).
    """
    load_tip = None if tip in (None, "override") else tip
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = load_plasma_data(data_dir, run_num, tip=load_tip)
    ne_arr = _load_ne(data_dir, run_num, load_tip, ne_arr, calibrated)
    xpos, *_ = load_sweep_axes(data_dir, run_num, tip=load_tip)

    ndx = tndx_list if tndx_list is not None else list(
        range(0, Vp_arr.shape[1], max(1, Vp_arr.shape[1] // 4)))

    name = f"{run_num}{tip_tag(load_tip)}-line"
    plot_iv_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, ndx,
                 save_fig=_resolve_save(save_fig, name), show=show,
                 title=title, isat=isat, fft=fft)


def plot_iv_line_run(ifn, tndx_list=None, save_fig=True, show=False,
                     calibrated=True):
    """Draw the IV line-scan plot(s) for a run from already-saved ``.npz``.

    Plots each tip :func:`Jun2026_IV.process_run` saved (see
    :func:`discover_tips`) from its ``-tip<T>-*.npz`` files -- no reprocessing
    (apart from the small Isat reference trace).  Run
    :func:`Jun2026_IV.process_run` first to create the ``.npz``.  ``save_fig``
    True writes the PNG via :func:`fig_path`; pass a path to override or False to
    skip.  ``calibrated`` True (default) plots interferometer-calibrated ne when
    available, falling back to raw ne with a note otherwise; pass False to force
    the raw ne.
    """
    data_dir = os.path.dirname(ifn)
    run_num = run_num_of(ifn)
    tips = discover_tips(ifn)

    # Everything tip-invariant is read ONCE here (the HDF5 is multi-GB): the
    # open run, the descriptive title ("<run_num>-<puff voltage ... ms>"), the
    # Isat reference trace (not in the saved .npz), and the saved all-shot FFT.
    # Each tip's plot then only loads its own npz.
    run = open_lapd(ifn)
    title = run_title(ifn, run_num, run=run)
    load_tip = None if tips == ["override"] else tips[0]
    _, _, npos, nshot = load_sweep_axes(data_dir, run_num, tip=load_tip)
    isat = _read_isat_trace(run, npos, nshot)
    fft = _read_isat_fft(ifn)

    for tip in tips:
        print(f"Plotting IV line-scan for tip {tip} from saved data...")
        _plot_iv_line_tip(data_dir, run_num, tip, tndx_list=tndx_list,
                          save_fig=save_fig, show=show,
                          title=title, isat=isat, fft=fft,
                          calibrated=calibrated)


# =========================================================================== #
#  Figure: combined IV line-scan (one run) + Isat raw/FFT (another run)
#
#  Six panels on one figure for comparing a swept-tip line scan against the
#  fixed-bias Isat fluctuations -- typically a different run for each:
#    1-4. Vp / Te / ne / Te*ne vs x at selected sweep times, from the IV run's
#         saved .npz, with the sweep instants scattered onto panel 5.
#    5.   The Isat run's raw fixed-bias trace, first `isat_nshot` shots averaged,
#         over 0..isat_tmax_ms ms (read with Jun2026_Isat.get_isat_at_position).
#    6.   That channel's all-shot-averaged FFT for the Isat run, loaded from the
#         batch npz (Jun2026_Isat.OUT_NPZ), limited to < fft_fmax_khz.
#  Reuses the IV loaders (load_plasma_data / load_sweep_axes), the Isat reader (jis)
#  and run_title -- no read/FFT logic is re-implemented here.
# =========================================================================== #

def puff_title(ifn, run=None):
    """The gas-puff setting of a run's description (``"Puff voltage 75V for 30ms"``).

    :func:`run_title` without the leading ``"<run_num>-"`` -- used as the
    combined figure's title.  Falls back to the bare run number when the
    description has no puff line, and ``None`` when it can't be read.
    """
    desc = _run_description(ifn, run=run)
    if desc is None:
        return None
    puff = parse_gas_puff(desc.raw)
    return _puff_label(puff) if puff else run_num_of(ifn)


def _read_isat_avg(run, scope_name, chan, isat_nshot):
    """Run's raw Isat trace averaged over the first ``isat_nshot`` shots.

    Reads just the first ``isat_nshot`` shots off disk (a positional shot slice
    -- the probe is stationary, so the leading shots are repeats at one position)
    and averages them.  ``run`` is the already-open run.  Returns
    ``(tarr, I_avg)`` in (seconds, signal).
    """
    Istack, tarr = run.channel(chan, scope_name=scope_name,
                               shots=slice(0, isat_nshot))
    return tarr, Istack.mean(axis=0)


def plot_iv_isat_combined(iv_ifn, isat_ifn, iv_tip=None, tndx_list=None,
                          isat_scope=jis.SCOPE_NAME, isat_chan=jis.CHAN,
                          isat_nshot=10, isat_tmax_ms=6.0,
                          fft_npz=None, fft_fmax_khz=80.0,
                          save_fig=False, show=True, calibrated=True):
    """Draw the combined 6-panel IV-line-scan + Isat figure (see section header).

    ``iv_ifn`` / ``isat_ifn`` are the run HDF5 paths for the IV line scan
    (panels 1-4, loaded from its saved .npz) and the Isat trace/FFT (panels 5-6).
    ``iv_tip`` selects the tip whose .npz to load (e.g. ``"L"``).  ``fft_npz``
    defaults to ``<isat dir>/Jun2026_Isat.OUT_NPZ``; the Isat run must be one of
    the runs in that batch npz.  The figure title is the gas-puff
    voltage/time from the IV run's description.  ``save_fig`` False -> don't save;
    ``show`` True pops the interactive window (set False for headless saving).
    ``calibrated`` True (default) uses interferometer-calibrated ne when
    available, else raw ne with a note.
    """
    data_dir = os.path.dirname(iv_ifn)
    run_num = run_num_of(iv_ifn)
    isat_run_num = run_num_of(isat_ifn)
    tndx_list = tndx_list if tndx_list is not None else [-7, -5, -3, -1]

    # Panels 1-3: IV arrays + axes from the IV run's saved .npz.
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = load_plasma_data(data_dir, run_num, tip=iv_tip)
    ne_arr = _load_ne(data_dir, run_num, iv_tip, ne_arr, calibrated)
    xpos, *_ = load_sweep_axes(data_dir, run_num, tip=iv_tip)

    # Panels 4-5: raw Isat (shot-averaged) + the all-shot-averaged FFT loaded
    # from the batch npz (shared loader, same as the line-scan figure).
    isat_run = open_lapd(isat_ifn)
    t_isat, I_avg = _read_isat_avg(isat_run, isat_scope, isat_chan, isat_nshot)
    fft = _read_isat_fft(isat_ifn, fft_npz)

    fig, axs = plt.subplots(6, 1, figsize=(12, 18))

    # Panels 1-4 share the IV line-scan drawing with plot_iv_line (Vp/Te/ne and
    # the Te*ne product). Title = the gas-puff voltage/time from the IV run's
    # description.  marked_times = the (colour, time) of each IV sweep instant,
    # scattered onto the Isat trace below.
    marked_times = _draw_iv_panels(axs[:4], Vp_arr, Te_arr, ne_arr, xpos, t_ls,
                                   tndx_list, puff_title(iv_ifn))

    # Panel 5: Isat raw, shot-averaged, 0..isat_tmax_ms; y clipped to 0..1.  The
    # IV sweep instants are scattered on the line, coloured to match the IV panels.
    _draw_isat_panel(axs[4], (t_isat, I_avg), marked_times,
                     tmax_ms=isat_tmax_ms, ylim=(0, 1))

    # Panel 6: all-shot-averaged FFT for the Isat run, < fft_fmax_khz.
    _draw_fft_panel(axs[5], fft, fft_fmax_khz,
                    f"Run {isat_run_num} Isat averaged FFT (all shots), "
                    f"scope '{isat_scope}' {isat_chan}",
                    chan_label=f"'{isat_scope}'/{isat_chan}")

    name = f"{run_num}{tip_tag(iv_tip)}-{isat_run_num}-combined"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, name), show=show,
                    compact_axes=axs[:4])


# =========================================================================== #
#  Figure: cross-correlation xy-plane map for a channel pair
#
#  Jun2026_xcorr.batch_xcorr stores one ensemble coherence/cross-phase spectrum
#  per (x, y) probe position (the Smith-1974 FFT estimator).  This section
#  collapses each to a per-position scalar and draws two imshow maps over the xy
#  plane (coherence, cross-phase) -- the only xcorr figure kept here.  (The
#  time-lag cross-correlation / Blackman-Tukey correlogram is no longer part of
#  the batch; the notebook can still call the analysis in Jun2026_xcorr directly
#  if a single-position spectrum or lag is wanted.)
# =========================================================================== #

def _load_xcorr_run(ifn, ch_a, ch_b, npz_path=None):
    """Load a pair's per-position xcorr result from the run's co-located npz.

    Returns ``(freq, gamma2, phase, pos_x, pos_y)`` where ``gamma2``/``phase`` are
    ``(npos, nf)`` (one ensemble spectrum per probe position, from
    :func:`Jun2026_xcorr.batch_xcorr`); ``freq`` is the shared axis and
    ``pos_x``/``pos_y`` are each position's (x, y).  Returns ``None`` (after
    printing why) if the npz or this pair's entry is missing, so callers can skip
    drawing.
    """
    if npz_path is None:
        npz_path = jxc.xcorr_npz_path(ifn)
    key = jxc._pair_key(ch_a, ch_b)
    try:
        with np.load(npz_path) as d:
            return (d["freq"], d[f"{key}__gamma2"], d[f"{key}__phase"],
                    d["pos_x"], d["pos_y"])
    except (OSError, KeyError) as e:
        print(f"  (xcorr: no saved entry for pair '{key}' in {npz_path} -- {e})")
        return None


def _draw_plane_maps(fig, axs, panels, extent):
    """Draw a row of xy-plane ``imshow`` maps with colorbars and cm axes.

    ``panels`` is a list of ``(grid, label, cmap, (vmin, vmax))`` (one per
    ``imshow``), drawn onto ``axs`` in order; shared by the xcorr plane figures so
    every map gets the same ``origin="lower"``, ``extent``, colorbar, and
    X/Y-position labelling.
    """
    for ax, (grid, label, cmap, (vmin, vmax)) in zip(axs, panels):
        im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("X Position [cm]")
        ax.set_ylabel("Y Position [cm]")
        ax.set_title(label)


def plot_xcorr_plane_run(ifn, ch_a=jxc.CH_A, ch_b=jxc.CH_B, npz_path=None,
                         fmin_khz=0.0, fmax_khz=80.0, gamma2_floor=0.2,
                         save_fig=True, show=False):
    """Draw coherence + cross-phase as xy-plane ``imshow`` maps.

    For a plane run, :func:`Jun2026_xcorr.batch_xcorr` stores one ensemble
    coherence/cross-phase spectrum (the Smith-1974 FFT estimator) per (x, y)
    position.  This collapses each to a per-position scalar and lays it on the xy
    grid (:func:`data_analysis.viz.plot_utils.grid_by_position`), drawing two maps
    side by side:

    * mean coherence ``gamma2`` (0..1) over the ``[fmin_khz, fmax_khz]`` band, and
    * mean cross-phase ``Delta-phi`` (deg) over that band.

    Cross-phase is only trustworthy where the channels are coherent, so positions
    with band coherence ``< gamma2_floor`` are blanked (NaN) in the phase map.
    ``save_fig`` True routes through :func:`fig_path`.
    """
    loaded = _load_xcorr_run(ifn, ch_a, ch_b, npz_path)
    if loaded is None:
        return
    freq, gamma2, phase, pos_x, pos_y = loaded

    # Per-position scalars over the frequency band: coherence is a plain mean;
    # cross-phase is averaged as a unit vector (angle of <e^{i phi}>) so the wrap
    # at +-180 deg is handled instead of a raw mean jumping across it.
    band = (freq >= fmin_khz * 1e3) & (freq <= fmax_khz * 1e3)
    g2_band = np.nanmean(gamma2[:, band], axis=1)
    ph_band = np.degrees(np.angle(np.nanmean(np.exp(1j * phase[:, band]), axis=1)))

    # Blank phase where the channels aren't coherent (it's noise there) so the map
    # shows structure only in the coherent region.
    ph_band = np.where(g2_band < gamma2_floor, np.nan, ph_band)

    g2_grid, extent = grid_by_position(pos_x, pos_y, g2_band)
    ph_grid, _ = grid_by_position(pos_x, pos_y, ph_band)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    _draw_plane_maps(fig, axs, [
        (g2_grid, r"coherence $\gamma^2$", "viridis", (0, 1)),
        (ph_grid, r"cross-phase $\Delta\phi$ [deg]", "twilight", (-180, 180)),
    ], extent)

    title = run_title(ifn, run_num_of(ifn)) or f"{run_num_of(ifn)} xcorr"
    fig.suptitle(f"{title}  —  {ch_a[0]}/{ch_a[1]} vs {ch_b[0]}/{ch_b[1]} "
                 f"({fmin_khz:g}-{fmax_khz:g} kHz band)",
                 fontsize=12, fontweight="bold")
    name = f"{run_num_of(ifn)}-xcorr-plane-{ch_a[0]}{ch_a[1]}-{ch_b[0]}{ch_b[1]}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, name), show=show)


def _load_xcorr_band_run(ifn, ch_a, ch_b, npz_path=None):
    """Load a pair's per-position narrow-band scalar coherence/phase from the npz.

    Reads what :func:`Jun2026_xcorr.batch_xcorr_band` wrote: one scalar coherence,
    one scalar cross-phase (radians), and the band-center frequency per probe
    position.  Returns ``(gamma2, phase, fpeak, nshots, fband, pos_x, pos_y)``
    where ``gamma2``/``phase``/``fpeak``/``nshots`` are ``(npos,)``, ``fpeak`` is
    the per-position band center in Hz (tracked peak or fixed band center), and
    ``fband`` is the ``(f_lo, f_hi)`` search/fixed window in Hz.  ``fpeak`` falls
    back to all-NaN for older npz files written before peak-tracking.  Returns
    ``None`` (after printing why) if the npz or this pair's band entry is missing.
    """
    if npz_path is None:
        npz_path = jxc.xcorr_npz_path(ifn)
    key = jxc._pair_key(ch_a, ch_b)
    try:
        with np.load(npz_path) as d:
            gamma2 = d[f"{key}__band_gamma2"]
            fpeak = (d[f"{key}__band_fpeak"] if f"{key}__band_fpeak" in d.files
                     else np.full(gamma2.shape, np.nan))
            return (gamma2, d[f"{key}__band_phase"], fpeak,
                    d[f"{key}__band_nshots"], d[f"{key}__band_fband"],
                    d["pos_x"], d["pos_y"])
    except (OSError, KeyError) as e:
        print(f"  (xcorr band: no saved entry for pair '{key}' in {npz_path} -- {e})")
        return None


def plot_xcorr_band_plane_run(ifn, ch_a=jxc.CH_A, ch_b=jxc.CH_B, npz_path=None,
                              gamma2_floor=0.2, save_fig=True, show=False):
    """Draw narrow-band coherence + phase-difference (+ peak-freq) xy-plane maps.

    Uses the per-position scalars from :func:`Jun2026_xcorr.batch_xcorr_band` (one
    band-averaged coherence + cross-phase per (x, y) position, collapsed the
    statistically correct way over the band's complex spectra).  Draws
    (:func:`data_analysis.viz.plot_utils.grid_by_position`):

    * band coherence ``gamma2`` (0..1),
    * band cross-phase ``Delta-phi`` (deg) -- the phase-difference map, and
    * (peak-tracking runs only) the tracked peak frequency ``f_peak`` (kHz) --
      how the mode frequency moves across the plane.

    The ``f_peak`` panel is added only when the per-position band center varies
    (a ``track_peak=True`` batch); a fixed-band batch has a constant center and
    draws just the two maps.  Cross-phase (and ``f_peak``) are only trustworthy
    where the channels are coherent, so positions with band coherence
    ``< gamma2_floor`` are blanked (NaN) in those maps.  ``save_fig`` True routes
    through :func:`fig_path`.
    """
    loaded = _load_xcorr_band_run(ifn, ch_a, ch_b, npz_path)
    if loaded is None:
        return
    gamma2, phase, fpeak, _nshots, fband, pos_x, pos_y = loaded
    fmin_khz, fmax_khz = fband[0] * 1e-3, fband[1] * 1e-3

    incoherent = gamma2 < gamma2_floor
    ph_deg = np.where(incoherent, np.nan, np.degrees(phase))

    g2_grid, extent = grid_by_position(pos_x, pos_y, gamma2)
    ph_grid, _ = grid_by_position(pos_x, pos_y, ph_deg)

    panels = [(g2_grid, r"coherence $\gamma^2$", "viridis", (0, 1)),
              (ph_grid, r"cross-phase $\Delta\phi$ [deg]", "twilight", (-180, 180))]

    # Peak-frequency map: only when the tracked center actually varies across the
    # plane (a fixed-band batch stores a constant center -- no map to draw).
    fp_khz = np.where(incoherent, np.nan, fpeak * 1e-3)
    peak_varies = np.nanstd(fp_khz) > 0
    if peak_varies:
        fp_grid, _ = grid_by_position(pos_x, pos_y, fp_khz)
        vlo, vhi = np.nanmin(fp_khz), np.nanmax(fp_khz)
        panels.append((fp_grid, r"peak $f$ [kHz]", "plasma", (vlo, vhi)))

    fig, axs = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6))
    _draw_plane_maps(fig, np.atleast_1d(axs), panels, extent)

    band_desc = (f"peak in {fmin_khz:g}-{fmax_khz:g} kHz"
                 if peak_varies else f"{fmin_khz:g}-{fmax_khz:g} kHz band")
    title = run_title(ifn, run_num_of(ifn)) or f"{run_num_of(ifn)} xcorr"
    fig.suptitle(f"{title}  —  {ch_a[0]}/{ch_a[1]} vs {ch_b[0]}/{ch_b[1]} "
                 f"({band_desc})",
                 fontsize=12, fontweight="bold")
    name = (f"{run_num_of(ifn)}-xcorr-band-plane-"
            f"{ch_a[0]}{ch_a[1]}-{ch_b[0]}{ch_b[1]}")
    finalize_figure(fig, save_fig=_resolve_save(save_fig, name), show=show)


# =========================================================================== #
#  Add further Jun-2026 figure sections below, each following the IV pattern:
#    plot_<fig>(...)            -- draw one figure, ending in finalize_figure()
#    plot_<fig>_run(ifn, ...)   -- load the saved .npz and call plot_<fig>
# =========================================================================== #


if __name__ == '__main__':


    iv_ifn = r"D:\data\LAPD\jun2026-jia\06-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"
    # isat_ifn = r"D:\data\LAPD\jun2026-jia\07-He-800G-bias40V-Isat-p29-plane_2026-06-10.hdf5"
    plot_iv_line_run(iv_ifn, save_fig=True, show=False, calibrated=True)
