"""Jun-2026 LAPD plotting -- shared figure module.

The **plotting** home for the Jun-2026 LAPD analyses.  The processing modules
(e.g. ``Jun2026_IV.py``) read raw HDF5, do the heavy lifting, and save results
to ``.npz``; this module reads those saved arrays back and draws the figures.
It does **no** analysis of its own; it opens the raw HDF5 only for the run
description (titles) and a few shots of the Isat reference trace
(``_read_isat_trace`` / ``_read_isat_avg``).

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
  coherence and cross-phase at a single frequency as side-by-side imshow maps
  (Smith-1974).
* ``plot_turbulence_steepening`` -- 4-panel nonlinear wave steepening vs.
  symmetric hollow-wall curvature figure from ``turbulence_transport_jun2026.npz``.
* ``plot_hollow_profiles`` -- clean dual-axis hollow-column ``n_e(x)`` and
  ``T_e(x)`` radial profile plot from ``profiles_summary_jun2026.npz``.
* ``plot_em_instability_overview`` -- 4-panel electromagnetic instability
  overview (raw fixed-bias LP ``Isat``, simultaneous raw B-dot, FFT of both,
  and reference-locked 2-D ``J_par`` & ``B_perp`` with shot probe markers).
* ``plot_sheath_closure`` -- 2-panel Port-35 conducting-sheath current closure
  figure (full bias-pulse ``I_2"`` & ``I_4"`` traces + zoomed 180-deg return
  current window).
* ``animate_mode_propagation`` -- side-by-side GIF animating measured 2-D
  ``J_par`` & ``B_perp`` in the lab frame vs. co-rotating ``E x B`` plasma
  rest frame.
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from data_analysis.io import open_lapd, parse_gas_puff
from data_analysis.plasma.langmuir import (
    load_plasma_data, load_sweep_axes, discover_channels, ne_is_calibrated)
from data_analysis.signal.core import downsample_blockmean
from data_analysis.utils import run_num_of
from data_analysis.viz import plot_utils
from data_analysis.viz.plot_utils import finalize_figure, grid_by_position

import Jun2026_Isat as jis
import Jun2026_xcorr as jxc

# Default subdirectory under <OUTPUT_ROOT>/figures/ (io/paths.py) for Jun-2026 figures.
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


# =========================================================================== #
#  Figure: IV line-scan (Vp / Te / ne vs x + Isat reference)
# =========================================================================== #

def _draw_iv_panels(axs4, Vp_arr, Te_arr, ne_arr, xpos, t_ls, tndx_list, title,
                    ne_unit="cm$^{-3}$"):
    """Draw the Vp / Te / ne / Te*ne vs x panels at the selected sweep times.

    ``axs4`` is the four axes to draw into (Vp, Te, ne, Te*ne, top to bottom).
    The fourth panel is the pressure-like product Te*ne derived from the Te and
    ne arrays (panels 2 and 3).  Each ``tndx_list`` time index is plotted in a
    rainbow colour across all four panels; the title goes on the top panel and
    the x-label on the bottom one.  ``ne_unit`` labels the ne (and Te*ne)
    panel: the default fits the interferometer-calibrated array; pass
    ``"A/cm$^2$"`` when plotting the raw ``analyze_IV`` proxy.  Returns
    ``marked_times`` -- the ``(colour, time_ms)`` of each drawn time, so a
    caller (the IV line-scan's Isat panel) can mark the same instants
    elsewhere.  Shared by :func:`plot_iv_line` and :func:`plot_iv_isat_combined`
    so the line-scan styling lives in one place.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(tndx_list)))
    Tene_arr = Te_arr * ne_arr   # pressure-like product, derived from panels 2-3
    panels = [(axs4[0], Vp_arr, "Vp [V]", "o-"),
              (axs4[1], Te_arr, "Te [eV]", "s-"),
              (axs4[2], ne_arr, f"ne [{ne_unit}]", "^-"),
              (axs4[3], Tene_arr, f"Te*ne [eV {ne_unit}]", "D-")]

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
                 fft_fmax_khz=80.0, ne_unit="cm$^{-3}$"):
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
                                   tndx_list, title, ne_unit=ne_unit)
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


def _plot_iv_line_tip(data_dir, run_num, tip, tndx_list=None, save_fig=True,
                      show=False, title=None, isat=None, fft=None):
    """Load one channel's saved IV arrays and draw the line-scan plot (no reprocessing).

    ``tip`` is a channel label from
    :func:`data_analysis.plasma.langmuir.discover_channels`.  ``title`` /
    ``isat`` / ``fft`` are the tip-invariant pieces the run driver reads once
    (descriptive title, Isat reference trace, saved FFT); each may be ``None``
    for a generic title / blank panel.  ``save_fig`` True routes the PNG through
    :func:`fig_path`; pass a path to override, or False to skip saving.

    ``ne`` is [cm^-3] whether or not the run was interferometer-calibrated; the
    subtitle says which, since the two differ in how far the absolute scale can
    be trusted, not in unit.
    """
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = load_plasma_data(data_dir, run_num, tip=tip)
    xpos, *_ = load_sweep_axes(data_dir, run_num, tip=tip)
    if not ne_is_calibrated(data_dir, run_num, tip=tip):
        title = f"{title or ''} [ne uncalibrated]".strip()

    ndx = tndx_list if tndx_list is not None else list(
        range(0, Vp_arr.shape[1], max(1, Vp_arr.shape[1] // 4)))

    plot_iv_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, ndx,
                 save_fig=_resolve_save(save_fig, f"{run_num}-{tip}-line"),
                 show=show, title=title, isat=isat, fft=fft,
                 ne_unit="cm$^{-3}$")


def plot_iv_line_run(ifn, tndx_list=None, save_fig=True, show=False):
    """Draw the IV line-scan plot(s) for a run from already-saved ``.npz``.

    Plots every channel the run's sweep npz holds (see
    :func:`data_analysis.plasma.langmuir.discover_channels`) -- no reprocessing
    apart from the small Isat reference trace.  Run
    :func:`Jun2026_IV.process_run` first to create the ``.npz``.  ``save_fig``
    True writes the PNG via :func:`fig_path`; pass a path to override or False
    to skip.
    """
    data_dir = os.path.dirname(ifn)
    run_num = run_num_of(ifn)
    tips = discover_channels(ifn)

    # Everything tip-invariant is read ONCE here (the HDF5 is multi-GB): the
    # open run, the descriptive title ("<run_num>-<puff voltage ... ms>"), the
    # Isat reference trace (not in the saved .npz), and the saved all-shot FFT.
    # Each tip's plot then only loads its own keys.
    run = open_lapd(ifn)
    title = run_title(ifn, run_num, run=run)
    _, _, npos, nshot = load_sweep_axes(data_dir, run_num, tip=tips[0])
    isat = _read_isat_trace(run, npos, nshot)
    fft = _read_isat_fft(ifn)

    for tip in tips:
        print(f"Plotting IV line-scan for {tip} from saved data...")
        _plot_iv_line_tip(data_dir, run_num, tip, tndx_list=tndx_list,
                          save_fig=save_fig, show=show,
                          title=title, isat=isat, fft=fft)


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
                          save_fig=False, show=True):
    """Draw the combined 6-panel IV-line-scan + Isat figure (see section header).

    ``iv_ifn`` / ``isat_ifn`` are the run HDF5 paths for the IV line scan
    (panels 1-4, loaded from its saved .npz) and the Isat trace/FFT (panels 5-6).
    ``iv_tip`` selects the channel whose arrays to load (e.g. ``"P29-L"``);
    ``None`` picks the only one.  ``fft_npz`` defaults to
    ``<isat dir>/Jun2026_Isat.OUT_NPZ``; the Isat run must be one of the runs in
    that batch npz.  The figure title is the gas-puff voltage/time from the IV
    run's description.  ``save_fig`` False -> don't save; ``show`` True pops the
    interactive window (set False for headless saving).
    """
    data_dir = os.path.dirname(iv_ifn)
    run_num = run_num_of(iv_ifn)
    isat_run_num = run_num_of(isat_ifn)
    tndx_list = tndx_list if tndx_list is not None else [-7, -5, -3, -1]

    # Panels 1-3: IV arrays + axes from the IV run's saved .npz.
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = load_plasma_data(data_dir, run_num, tip=iv_tip)
    xpos, *_ = load_sweep_axes(data_dir, run_num, tip=iv_tip)
    puff = puff_title(iv_ifn)
    if not ne_is_calibrated(data_dir, run_num, tip=iv_tip):
        puff = f"{puff} [ne uncalibrated]"

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
                                   tndx_list, puff, ne_unit="cm$^{-3}$")

    # Panel 5: Isat raw, shot-averaged, 0..isat_tmax_ms; y clipped to 0..1.  The
    # IV sweep instants are scattered on the line, coloured to match the IV panels.
    _draw_isat_panel(axs[4], (t_isat, I_avg), marked_times,
                     tmax_ms=isat_tmax_ms, ylim=(0, 1))

    # Panel 6: all-shot-averaged FFT for the Isat run, < fft_fmax_khz.
    _draw_fft_panel(axs[5], fft, fft_fmax_khz,
                    f"Run {isat_run_num} Isat averaged FFT (all shots), "
                    f"scope '{isat_scope}' {isat_chan}",
                    chan_label=f"'{isat_scope}'/{isat_chan}")

    name = f"{run_num}-{iv_tip or 'all'}-{isat_run_num}-combined"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, name), show=show,
                    compact_axes=axs[:4])


# =========================================================================== #
#  Figure: cross-correlation xy-plane map for a channel pair
#
#  Jun2026_xcorr.batch_xcorr stores one ensemble coherence/cross-phase spectrum
#  per (x, y) probe position (the Smith-1974 FFT estimator).  This section picks
#  ONE frequency, samples each position's spectrum at that bin, and draws two
#  imshow maps over the xy plane (coherence, cross-phase) -- the only xcorr figure
#  kept here.  (The time-lag cross-correlation / Blackman-Tukey correlogram is no
#  longer part of the batch; the notebook can still call the analysis in
#  Jun2026_xcorr directly if a single-position spectrum or lag is wanted.)
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
            # Prefer per-pair coordinates (co-moving probes in a two-drive run
            # can scan in opposite order); fall back to file-level for older npz.
            pos_x = d[f"{key}__pos_x"] if f"{key}__pos_x" in d.files else d["pos_x"]
            pos_y = d[f"{key}__pos_y"] if f"{key}__pos_y" in d.files else d["pos_y"]
            return (d["freq"], d[f"{key}__gamma2"], d[f"{key}__phase"],
                    pos_x, pos_y)
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
                         freq_khz=10.0, gamma2_floor=0.2,
                         save_fig=True, show=False):
    """Draw coherence + cross-phase as xy-plane ``imshow`` maps at ONE frequency.

    For a plane run, :func:`Jun2026_xcorr.batch_xcorr` stores one ensemble
    coherence/cross-phase spectrum (the Smith-1974 FFT estimator) per (x, y)
    position.  This samples each position's spectrum at the single frequency bin
    nearest ``freq_khz`` and lays that value on the xy grid
    (:func:`data_analysis.viz.plot_utils.grid_by_position`), drawing two maps side
    by side:

    * coherence ``gamma2`` (0..1) at ``freq_khz``, and
    * cross-phase ``Delta-phi`` (deg) at ``freq_khz``.

    No frequency averaging is done -- the maps are the plane structure of a single
    spectral line.  ``freq_khz`` is snapped to the nearest available bin (the
    actual bin used is reported in the title).  Cross-phase is only trustworthy
    where the channels are coherent, so positions with coherence ``< gamma2_floor``
    at that frequency are blanked (NaN) in the phase map.  ``save_fig`` True routes
    through :func:`fig_path`.
    """
    loaded = _load_xcorr_run(ifn, ch_a, ch_b, npz_path)
    if loaded is None:
        return
    freq, gamma2, phase, pos_x, pos_y = loaded

    # Pick the single frequency bin nearest freq_khz; sample every position's
    # spectrum there (no averaging). fbin is the actual bin frequency used.
    fi = int(np.argmin(np.abs(freq - freq_khz * 1e3)))
    fbin_khz = freq[fi] * 1e-3

    # TESTING (hard-coded): coherence is averaged over the center bin +- 2 bins
    # (5 bins total, clipped to the array edges); phase stays at the single center
    # bin. Remove / parameterize this once the comparison is settled.
    _NBIN_HALF = 2
    lo = max(fi - _NBIN_HALF, 0)
    hi = min(fi + _NBIN_HALF + 1, freq.size)
    g2_f = np.nanmean(gamma2[:, lo:hi], axis=1)
    ph_f = np.degrees(phase[:, fi])

    # Blank phase where the channels aren't coherent at this frequency (it's noise
    # there) so the map shows structure only in the coherent region.  Gate on the
    # single-bin coherence so the phase floor matches its own frequency.
    ph_f = np.where(gamma2[:, fi] < gamma2_floor, np.nan, ph_f)

    g2_grid, extent = grid_by_position(pos_x, pos_y, g2_f)
    ph_grid, _ = grid_by_position(pos_x, pos_y, ph_f)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    _draw_plane_maps(fig, axs, [
        (g2_grid, rf"coherence $\gamma^2$ ($\pm${_NBIN_HALF} bins)",
         "viridis", (0, 1)),
        (ph_grid, r"cross-phase $\Delta\phi$ [deg]", "twilight", (-180, 180)),
    ], extent)

    title = run_title(ifn, run_num_of(ifn)) or f"{run_num_of(ifn)} xcorr"
    fig.suptitle(f"{title}  —  {jxc.pair_label(ch_a, ch_b)} "
                 f"(f = {fbin_khz:.3g} kHz)",
                 fontsize=12, fontweight="bold")
    name = (f"{run_num_of(ifn)}-xcorr-plane-{ch_a[0]}{ch_a[1]}-{ch_b[0]}{ch_b[1]}"
            f"-{fbin_khz:.3g}kHz")
    finalize_figure(fig, save_fig=_resolve_save(save_fig, name), show=show)


# =========================================================================== #
#  Add further Jun-2026 figure sections below, each following the IV pattern:
#    plot_<fig>(...)            -- draw one figure, ending in finalize_figure()
#    plot_<fig>_run(ifn, ...)   -- load the saved .npz and call plot_<fig>
# =========================================================================== #


def plot_turbulence_steepening(npz_path=None, *, data_dir=None,
                               save_fig=True, show=False):
    """4-panel nonlinear steepening & phase-folded sawtooth waveform figure.

    Reads ``turbulence_transport_jun2026.npz`` produced by
    ``Jun2026_turbulence.py`` (zero raw HDF5 reads) and draws:

    * **(a)** Phase-folded stationary ``Isat`` cycle waveforms vs. pure
      sinusoid ``cos(theta)`` across puff and outer-probe runs (``02``, ``03``,
      ``04``, ``27``).
    * **(b)** Run 32 hollow-wall ``[3.5, 4.5) cm`` phase-folded velocity
      (``M_r``, ``M_theta``) and flow-cancelled density (``Isat_0``) vs.
      ``cos(theta)`` — illustrating intrinsic velocity-field sawtooth
      steepening independent of ``n_0''(r)``.
    * **(c)** Radial biphase winding ``beta(r)`` and orthogonal decomposition
      into symmetric wall curvature ``(A2/A1)*cos(beta)`` vs. sawtooth
      steepening ``(A2/A1)*sin(beta)`` across ``r = 0..7 cm`` on Runs 11–12.
    * **(d)** Campaign summary of ``steepening`` vs. ``curvature`` across the
      Day-1 puff scan (Runs ``00–04``) and ``B0`` scan (Runs ``28, 27, 30``).
    """
    import Jun2026_turbulence as jtb

    path = npz_path or os.path.join(data_dir or jtb.DATA_DIR, jtb.OUT_NAME)
    with np.load(path, allow_pickle=True) as d:
        res = {k: d[k] for k in d.files}

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2))

    # --- (a) Phase-folded stationary Isat waveforms vs pure cos(theta) ---
    ax = axs[0, 0]
    th_ref = np.linspace(-np.pi, np.pi, 256)
    ax.plot(th_ref / (2.0 * np.pi), np.cos(th_ref), "k--", lw=1.6, alpha=0.65,
            label=r"Pure sine $\cos\theta$")
    specs_a = (
        ("02", "Run 02 (30 ms, inner wall)", "#d62728"),
        ("03", "Run 03 (35 ms, inner wall)", "#ff7f0e"),
        ("04", "Run 04 (40 ms, broadband)", "#7f7f7f"),
        ("27", "Run 27 (35 ms, outer probe)", "#1f77b4"),
    )
    for rnum, lbl, col in specs_a:
        p = f"spec_{rnum}__"
        if p + "folded_wave" not in res:
            continue
        ph = np.asarray(res[p + "folded_phase_rad"], dtype=float) / (2.0 * np.pi)
        w = np.asarray(res[p + "folded_wave"], dtype=float)
        ws = np.asarray(res[p + "folded_std"], dtype=float)
        beta = float(res[p + "biphase_deg"])
        slp = float(res[p + "slope_ratio_med"])
        rf = float(res[p + "rise_frac_med"])
        ax.plot(ph, w, lw=2.2, color=col,
                label=rf"{lbl}: $\beta={beta:+.0f}^\circ$, slp={slp:.2f}$\times$, $f_r={rf:.2f}$")
        ax.fill_between(ph, w - ws, w + ws, color=col, alpha=0.12)
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"Fundamental cycle phase $\theta / 2\pi$")
    ax.set_ylabel(r"Normalized waveform $\tilde{I}_{\mathrm{sat}} / A_1$")
    ax.set_title("(a) Phase-Folded Stationary $I_{\\mathrm{sat}}$ vs. Pure Sinusoid")
    ax.legend(fontsize=8.2, loc="lower center", framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # --- (b) Run 32 wall [3.5, 4.5) cm velocity (M_r, M_theta) vs density ---
    ax = axs[0, 1]
    ax.plot(th_ref / (2.0 * np.pi), np.cos(th_ref), "k--", lw=1.6, alpha=0.65,
            label=r"Pure sine $\cos\theta$")
    if "r32__steep_phase_rad" in res:
        ph32 = np.asarray(res["r32__steep_phase_rad"], dtype=float) / (2.0 * np.pi)
        bi = 2  # [3.5, 4.5) cm hollow-wall band
        for tag, lbl, col in (
            ("mr", r"Radial velocity $\tilde{M}_r$", "#d62728"),
            ("mt", r"Azimuthal velocity $\tilde{M}_\theta$", "#2ca02c"),
            ("isat0", r"Flow-cancelled $I_{\mathrm{sat},0}$", "#1f77b4"),
        ):
            w = np.asarray(res[f"r32__steep_{tag}_folded_waves"], dtype=float)[bi]
            beta = float(res[f"r32__steep_{tag}_biphase_deg"][bi])
            stp = float(res[f"r32__steep_{tag}_steepening_comp"][bi])
            slp = float(res[f"r32__steep_{tag}_slope_ratio"][bi])
            bkey = f"r32__steep_linear_tip_{tag}_steepening_comp"
            bstr = f" (lin-tip {float(res[bkey][bi]):+.2f})" if (bkey in res and np.isfinite(res[bkey][bi])) else ""
            ax.plot(ph32, w, lw=2.3, color=col,
                    label=rf"{lbl}: $\beta={beta:+.0f}^\circ$, sin={stp:+.2f}{bstr}, slp={slp:.2f}$\times$")
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"Fundamental cycle phase $\theta / 2\pi$")
    ax.set_ylabel(r"Normalized waveform $\tilde{x} / A_1$")
    ax.set_title(r"(b) Run 32 Wall $[3.5, 4.5)$ cm: Intrinsic Velocity vs. Density Steepening")
    ax.legend(fontsize=8.3, loc="lower center", framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # --- (c) Radial decomposition across hollow wall (Runs 11 & 12) ---
    ax = axs[1, 0]
    for rnum, puff, col, mk in (("11", 30, "#d62728", "o"), ("12", 35, "#1f77b4", "s")):
        pr = f"radst_{rnum}__"
        if pr + "r_cm" not in res:
            continue
        r = np.asarray(res[pr + "r_cm"], dtype=float)
        stp = np.asarray(res[pr + "steepening_comp"], dtype=float)
        crv = np.asarray(res[pr + "curvature_comp"], dtype=float)
        ax.plot(r, stp, color=col, marker=mk, lw=2.2, ms=6,
                label=rf"Run {rnum} ({puff} ms) steepening $(A_2/A_1)\sin\beta$")
        ax.plot(r, crv, color=col, marker=mk, lw=1.4, ls="--", ms=4.5, alpha=0.65,
                label=rf"Run {rnum} ({puff} ms) curvature $(A_2/A_1)\cos\beta$")
    ax.axhline(0.0, color="0.5", lw=0.9)
    ax.axvspan(2.5, 3.5, color="#ffcccc", alpha=0.25, label="Mid-wall (steep rise)")
    ax.axvspan(5.0, 6.2, color="#cce5ff", alpha=0.25, label="Outer shear (steep fall)")
    ax.set_xlabel(r"Radius $r = x - x_0$ [cm]")
    ax.set_ylabel(r"Harmonic component amplitude")
    ax.set_title(r"(c) Radial Decomposition Across Hollow Wall (Runs 11–12 $I_{\mathrm{sat},0}$)")
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # --- (d) Campaign summary: Puff scan (00-04) & B0 scan (28, 27, 30) ---
    ax = axs[1, 1]
    run_order = (
        ("00", "24ms\n800G"), ("01", "25ms\n800G"), ("02", "30ms\n800G"),
        ("03", "35ms\n800G"), ("04", "40ms\n800G"),
        ("28", "35ms\n600G*"), ("27", "35ms\n800G*"), ("30", "35ms\n1000G*"),
    )
    xs = np.arange(len(run_order))
    stps = [float(res[f"spec_{r}__steepening_comp"]) for r, _ in run_order]
    crvs = [float(res[f"spec_{r}__curvature_comp"]) for r, _ in run_order]
    slps = [float(res[f"spec_{r}__slope_ratio_med"]) for r, _ in run_order]
    w_bar = 0.36
    ax.bar(xs - w_bar / 2, stps, width=w_bar, color="#d62728", alpha=0.88,
           label=r"Sawtooth steepening $(A_2/A_1)\sin\beta$")
    ax.bar(xs + w_bar / 2, crvs, width=w_bar, color="#1f77b4", alpha=0.72,
           label=r"Symmetric curvature $(A_2/A_1)\cos\beta$")
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.axvline(4.5, color="0.4", lw=1.0, ls=":")
    for xi, sv, slp in zip(xs, stps, slps):
        if not (np.isfinite(sv) and np.isfinite(slp) and slp > 0.0):
            continue
        va = "bottom" if sv >= 0 else "top"
        off = 0.02 if sv >= 0 else -0.02
        txt = f"{slp:.1f}x↑" if slp >= 1.0 else f"{1.0/slp:.1f}x↓"
        ax.text(xi - w_bar / 2, sv + off, txt, ha="center", va=va, fontsize=7.8,
                fontweight="bold", color="#8b0000")
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in run_order], fontsize=8.5)
    ax.set_ylabel(r"2nd-harmonic component")
    ax.set_title(r"(d) Puff Scan & $B_0$ Scan (*outer probe; slope ratio annotated)")
    ax.set_ylim(-0.42, 0.56)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Jun-2026 Nonlinear Wave Steepening vs. Symmetric Wall Curvature",
                 fontsize=13.5, fontweight="bold")
    finalize_figure(fig, save_fig=_resolve_save(save_fig, "turbulence_steepening"), show=show)


# =========================================================================== #
#  Finding #1: Electromagnetic CWM mechanism figures & propagation GIF
# =========================================================================== #

def _run_hdf5(data_dir, run_num):
    """Locate the single HDF5 file for ``run_num`` inside ``data_dir``."""
    hits = sorted(glob.glob(os.path.join(data_dir, f"{run_num}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {run_num} in {data_dir}")
    if len(hits) > 1:
        raise ValueError(f"multiple HDF5 files for run {run_num}: {hits}")
    return hits[0]


def _shot_xy(pos, shot):
    """Return ``(x_cm, y_cm)`` for integer ``shot`` from a ``ProbePositions`` or array."""
    arr = pos.pos_array if hasattr(pos, "pos_array") else pos
    shots = np.asarray(arr["shot_num"], dtype=int)
    hits = np.flatnonzero(shots == int(shot))
    if hits.size == 0:
        raise ValueError(f"shot {shot} not found in motion positions array")
    i = int(hits[0])
    return float(arr["x"][i]), float(arr["y"][i])


def _windowed_fft_amp(sig, dt_s, mask, pad_factor=8):
    """Single-sided Hann-windowed FFT amplitude [same units as ``sig``] vs kHz."""
    y = np.asarray(sig[mask], dtype=float)
    y = y - float(np.mean(y))
    n = y.size
    win = np.hanning(n)
    nfft = n * int(pad_factor)
    spec = 2.0 * np.abs(np.fft.rfft(y * win, n=nfft)) / float(np.sum(win))
    freq_khz = np.fft.rfftfreq(nfft, d=float(dt_s)) * 1e-3
    return freq_khz, spec


def _load_bdot_2d_snapshot(bdot_npz_path, run_num):
    """Load complex ``h1`` 2-D ``(xs, ys, Bx, By, Jz, meta)`` from ``bdot_2d_jun2026.npz``."""
    with np.load(bdot_npz_path, allow_pickle=True) as bd:
        p = f"{run_num}__"
        xs = np.asarray(bd[p + "x_cm"], dtype=float)
        ys = np.asarray(bd[p + "y_cm"], dtype=float)
        Bx = np.asarray(bd[p + "h1__Bx"], dtype=complex)
        By = np.asarray(bd[p + "h1__By"], dtype=complex)
        Jz = np.asarray(bd[p + "h1__Jz"], dtype=complex)
        if Bx.shape == (xs.size, ys.size) and xs.size != ys.size:
            Bx, By, Jz = Bx.T, By.T, Jz.T
        meta = {
            "b0_g": float(bd[p + "b0_g"]),
            "f0_hz": float(bd[p + "f0_hz"]),
            "bz_over_bperp": float(bd[p + "h1__bz_over_bperp"]),
            "center_x_cm": float(bd[p + "h1__center_x_cm"]),
            "center_y_cm": float(bd[p + "h1__center_y_cm"]),
            "bdot_calibrated": bool(bd[p + "bdot_calibrated"]),
        }
    return xs, ys, Bx, By, Jz, meta


def plot_hollow_profiles(
    run_num="02",
    *,
    data_dir=None,
    profiles_npz=None,
    x_lim=(-9.5, 9.5),
    save_fig=True,
    show=False,
):
    """Clean dual-axis hollow-column ``n_e(x)`` and ``T_e(x)`` radial profiles.

    Reads ``profiles_summary_jun2026.npz`` produced by ``Jun2026_profiles.py``
    (zero raw HDF5 reads) and plots the pre-onset reference sweep's ``n_e``
    (blue) and ``T_e`` (red) vs. ``x - x_c`` with ``eta_e`` annotated.
    """
    import Jun2026_profiles as jpr

    ddir = data_dir or jpr.DATA_DIR
    prof_path = profiles_npz or os.path.join(ddir, jpr.OUT_NAME)
    with np.load(prof_path, allow_pickle=True) as pf:
        p = f"{run_num}__"
        x_cm = np.asarray(pf[p + "x_cm"], dtype=float)
        xc_cm = float(pf[p + "x_center_cm"])
        ir = int(pf[p + "ref_sweep_idx"])
        ne = np.asarray(pf[p + "ne"][:, ir], dtype=float) * 1e-12  # 10^18 m^-3
        te = np.asarray(pf[p + "Te"][:, ir], dtype=float)
        eta_e = float(pf[p + "ref_eta_e_steep"]) if (p + "ref_eta_e_steep") in pf.files else np.nan
        puff_ms = float(pf[p + "gas_puff_ms"]) if (p + "gas_puff_ms") in pf.files else np.nan

    dx = x_cm - xc_cm
    m_x = (dx >= float(x_lim[0])) & (dx <= float(x_lim[1]))

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    l1, = ax.plot(dx[m_x], ne[m_x], "o-", color="#1d4ed8", lw=2.2, ms=5.2, label=r"$n_e$")
    ax.set_xlabel(r"$x - x_c$ [cm]")
    ax.set_ylabel(r"$n_e$ [$10^{18}\;\mathrm{m}^{-3}$]", color="#1d4ed8")
    ax.tick_params(axis="y", labelcolor="#1d4ed8")
    ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    ax.set_ylim(0.0, max(float(np.nanmax(ne[m_x])) * 1.18, 17.0))
    ax.grid(alpha=0.22)

    ax2 = ax.twinx()
    l2, = ax2.plot(dx[m_x], te[m_x], "s-", color="#dc2626", lw=2.2, ms=5.0, label=r"$T_e$")
    ax2.set_ylabel(r"$T_e$ [eV]", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.set_ylim(0.0, max(float(np.nanmax(te[m_x])) * 1.24, 3.1))

    if np.isfinite(eta_e):
        ax.text(0.04, 0.06, rf"$\eta_e = {eta_e:.2f}$", transform=ax.transAxes,
                fontsize=10.5, va="bottom")
    ax.legend(handles=[l1, l2], loc="upper right", frameon=True, framealpha=0.92)
    puff_str = f", {puff_ms:g} ms" if np.isfinite(puff_ms) else ""
    ax.set_title(f"Profiles (Run {run_num}{puff_str})", fontweight="bold")

    save_name = "fig1a_profiles" if str(run_num) == "02" else f"fig1a_profiles_run{run_num}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, save_name), show=show)


def plot_em_instability_overview(
    run_num="26",
    shot=716,
    *,
    bdot_ch="C1",
    lp_ch="C2",
    matched_puff_run="03",
    data_dir=None,
    ifn=None,
    bdot_npz=None,
    modes_npz=None,
    t_window_ms=(-0.5, 5.5),
    fft_window_ms=(2.0, 4.8),
    fmax_khz=20.0,
    pad_factor=8,
    save_fig=True,
    show=False,
):
    """4-panel electromagnetic instability overview (default: Run 26, shot 716).

    * **(a)** Raw Langmuir ``Isat`` trace across the bias pulse.
    * **(b)** Simultaneous raw P28 B-dot trace (baseline-zeroed on pre-bias).
    * **(c)** Normalized Hann-windowed FFT magnitude of both over ``fft_window_ms``.
    * **(d)** Reference-locked 2-D ``Re[J_par]`` & ``B_perp`` from
      ``bdot_2d_jun2026.npz`` with Port-35 ``2"``/``4"`` footprints and the
      exact probe locations of ``shot``.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Circle
    import Jun2026_IV as jiv
    import Jun2026_bdot_2d as jbd
    import Jun2026_modes as jmd

    ddir = data_dir or jbd.DATA_DIR
    hdf5_path = ifn or _run_hdf5(ddir, run_num)
    bdot_path = bdot_npz or os.path.join(ddir, jbd.OUT_NAME)
    modes_path = modes_npz or os.path.join(ddir, jmd.OUT_NAME)

    run = open_lapd(hdf5_path)
    tl = np.asarray(run.time_array(scope_name="lpscope"), dtype=float) * 1e3
    tb = np.asarray(run.time_array(scope_name="bdotscope"), dtype=float) * 1e3
    dt_l = float(tl[1] - tl[0]) * 1e-3
    dt_b = float(tb[1] - tb[0]) * 1e-3

    st_lp, _ = run.channel(lp_ch, scope_name="lpscope", shots=[int(shot)])
    st_bd, _ = run.channel(bdot_ch, scope_name="bdotscope", shots=[int(shot)])
    lp_raw = np.asarray(st_lp[0], dtype=float)
    bd_raw = np.asarray(st_bd[0], dtype=float)

    pos_lp = jiv.read_lp_positions(hdf5_path, motion_group_name=jbd.LP_MOTION_GROUP)
    pos_bd = jiv.read_lp_positions(hdf5_path, motion_group_name=jbd.BDOT_MOTION_GROUP)
    lp_x, lp_y = _shot_xy(pos_lp, shot)
    bd_x, bd_y = _shot_xy(pos_bd, shot)

    # Zero pre-bias DC offset on B-dot (inductive coil has zero DC)
    m_pre_b = (tb >= t_window_ms[0]) & (tb <= -0.05)
    bd_zero = bd_raw - (float(np.mean(bd_raw[m_pre_b])) if np.any(m_pre_b) else 0.0)

    t_lo, t_hi = float(t_window_ms[0]), float(t_window_ms[1])
    w_lo, w_hi = float(fft_window_ms[0]), float(fft_window_ms[1])
    m_tl = (tl >= t_lo) & (tl <= t_hi)
    m_tb = (tb >= t_lo) & (tb <= t_hi)
    m_wl = (tl >= w_lo) & (tl <= w_hi)
    m_wb = (tb >= w_lo) & (tb <= w_hi)

    fl, sl = _windowed_fft_amp(lp_raw, dt_l, m_wl, pad_factor=pad_factor)
    fb, sb = _windowed_fft_amp(bd_zero, dt_b, m_wb, pad_factor=pad_factor)
    m_f0_l = (fl >= 2.5) & (fl <= 5.5)
    m_f0_b = (fb >= 2.5) & (fb <= 5.5)
    f0_khz = float(fl[m_f0_l][int(np.argmax(sl[m_f0_l]))])

    xs, ys, Bx, By, Jz, meta = _load_bdot_2d_snapshot(bdot_path, run_num)
    ph_max = np.angle(Jz[np.unravel_index(int(np.nanargmax(np.abs(Jz))), Jz.shape)])
    rot = np.exp(-1j * ph_max)
    Jz_r = np.real(Jz * rot)
    Bx_r = np.real(Bx * rot)
    By_r = np.real(By * rot)
    is_cal = bool(meta["bdot_calibrated"])

    beta_rat = np.nan
    db_over_b0 = np.nan
    if is_cal and np.isfinite(meta.get("b0_g", np.nan)) and meta["b0_g"] > 0.0:
        # Compute f0 time-RMS peak |dB_perp| / B0 directly from the plotted calibrated
        # h1 map (avoids any broadband/2f0 inflation from dividing std(dB/dt) by w0).
        bp_rms_t = np.sqrt(0.5 * (np.abs(Bx) ** 2 + np.abs(By) ** 2))
        db_over_b0 = float(np.nanmax(bp_rms_t)) / (float(meta["b0_g"]) * 1e-4)
    if os.path.exists(modes_path):
        with np.load(modes_path, allow_pickle=True) as md:
            bkey = f"disp__{matched_puff_run}__beta_e_over_me_mi"
            if bkey in md.files:
                beta_rat = float(md[bkey])

    ch_map = {"C1": r"$\dot{B}_x$", "C2": r"$\dot{B}_y$", "C3": r"$\dot{B}_z$"}
    b_lbl = ch_map.get(bdot_ch, bdot_ch)

    fig = plt.figure(figsize=(14.4, 7.6))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.18, 1.0], wspace=0.22, hspace=0.44)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[:, 1])

    # (a) Raw Langmuir Isat
    ax_a.plot(tl[m_tl], lp_raw[m_tl], color="#1d4ed8", lw=0.9)
    ax_a.axvspan(w_lo, w_hi, color="0.91", alpha=0.65, zorder=0)
    ax_a.set_ylabel(r"Langmuir $I_{\mathrm{sat}}$ [V]")
    ax_a.set_xlim(t_lo, t_hi)
    ax_a.grid(alpha=0.22)
    ax_a.set_title(f"(a) Raw Langmuir Probe (Run {run_num}, shot {shot})", fontweight="bold")

    # (b) Raw B-dot (symmetric y-limit scaled to active oscillation so turn-off spike doesn't squash)
    bd_mv = bd_zero[m_tb] * 1e3
    bd_win_p99 = float(np.nanpercentile(np.abs(bd_zero[m_wb] * 1e3), 99.5))
    ylim_b = max(1.5 * bd_win_p99, 100.0)
    ax_b.plot(tb[m_tb], bd_mv, color="#dc2626", lw=0.8)
    ax_b.axvspan(w_lo, w_hi, color="0.91", alpha=0.65, zorder=0)
    ax_b.set_xlabel("Time relative to bias turn-on [ms]")
    ax_b.set_ylabel(f"B-dot {b_lbl} [mV]")
    ax_b.set_xlim(t_lo, t_hi)
    ax_b.set_ylim(-ylim_b, ylim_b)
    ax_b.grid(alpha=0.22)
    ax_b.set_title("(b) Raw Magnetic Probe (P28 B-dot, same shot)", fontweight="bold")

    # (c) Normalized FFT magnitude
    m_fl = fl <= float(fmax_khz)
    m_fb = fb <= float(fmax_khz)
    sl_norm = sl[m_fl] / float(sl[m_f0_l].max())
    sb_norm = sb[m_fb] / float(sb[m_f0_b].max())

    ax_c.plot(fl[m_fl], sl_norm, color="#1d4ed8", lw=1.9, label=r"Langmuir $I_{\mathrm{sat}}$")
    ax_c.plot(fb[m_fb], sb_norm, color="#dc2626", lw=1.6, ls="--", label=f"B-dot {b_lbl}")
    for k, lbl in ((1, r"$f_0$"), (2, r"$2f_0$"), (3, r"$3f_0$"), (4, r"$4f_0$")):
        fk = k * f0_khz
        if fk > fmax_khz - 0.5:
            continue
        mkl = (fl[m_fl] >= fk - 0.5) & (fl[m_fl] <= fk + 0.5)
        mkb = (fb[m_fb] >= fk - 0.5) & (fb[m_fb] <= fk + 0.5)
        if not (np.any(mkl) and np.any(mkb)):
            continue
        yl, yb = float(sl_norm[mkl].max()), float(sb_norm[mkb].max())
        yk = max(yl, yb)
        f_pk = (float(fb[m_fb][mkb][int(np.argmax(sb_norm[mkb]))])
                if yb >= yl else float(fl[m_fl][mkl][int(np.argmax(sl_norm[mkl]))]))
        ax_c.annotate(
            f"{lbl} ({f_pk:.1f} kHz)" if k == 1 else lbl,
            xy=(f_pk, yk), xytext=(f_pk + 0.45, min(yk + 0.16, 1.13)),
            fontsize=9.3, fontweight="bold", color="0.15",
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9),
        )
    ax_c.set_xlabel("Frequency [kHz]")
    ax_c.set_ylabel("Normalized FFT")
    ax_c.set_xlim(0.0, float(fmax_khz))
    ax_c.set_ylim(0.0, 1.28)
    ax_c.grid(alpha=0.22)
    ax_c.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax_c.set_title(f"(c) FFT Magnitude ({w_lo:.1f}–{w_hi:.1f} ms shaded window)", fontweight="bold")

    # (d) 2-D J_par & B_perp with shot probe markers
    XX, YY = np.meshgrid(xs, ys)
    if is_cal:
        j_max_a = float(np.nanmax(np.abs(Jz_r)))
        j_scale, j_unit = (1e-3, r"kA/m$^2$") if j_max_a >= 1e3 else (1.0, r"A/m$^2$")
        j_plot = Jz_r * j_scale
        j_lim = j_max_a * j_scale
        im = ax_d.pcolormesh(XX, YY, j_plot, cmap="RdBu_r", vmin=-j_lim, vmax=j_lim, shading="auto")
        cb = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.03, shrink=0.84)
        cb.set_label(rf"$\mathrm{{Re}}[J_\parallel]$ [{j_unit}]")

        Bx_g, By_g = Bx_r * 1e4, By_r * 1e4
        bscale_g = float(np.nanpercentile(np.hypot(Bx_g, By_g), 95))
        if bscale_g > 0.0:
            from matplotlib.patches import FancyBboxPatch
            qv = ax_d.quiver(XX, YY, Bx_g, By_g, color="k", scale=bscale_g * 14.0, width=0.0042, alpha=0.82)
            key_g = 0.2 if bscale_g >= 0.15 else 0.1
            ax_d.add_patch(FancyBboxPatch(
                (0.72, 0.025), 0.25, 0.065,
                boxstyle="round,pad=0.012", transform=ax_d.transAxes,
                fc="white", ec="none", alpha=0.88, zorder=7,
            ))
            qk = ax_d.quiverkey(
                qv, 0.80, 0.058, key_g, f"{key_g:g} G",
                labelpos="E", coordinates="axes", fontproperties={"size": 9.4, "weight": "bold"},
            )
            qk.set_zorder(8)
    else:
        Jz_norm = Jz_r / float(np.nanmax(np.abs(Jz_r)))
        im = ax_d.pcolormesh(XX, YY, Jz_norm, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
        cb = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.03, shrink=0.84)
        cb.set_label(r"$\mathrm{Re}[J_\parallel] / J_{\max}$")

        bscale = float(np.nanpercentile(np.hypot(Bx_r, By_r), 95))
        if bscale > 0.0:
            ax_d.quiver(XX, YY, Bx_r / bscale, By_r / bscale, color="k", scale=14.0, width=0.0042, alpha=0.82)

    ax_d.add_patch(Circle((0.0, 0.0), 2.54, fill=False, ec="#16a34a", ls="-", lw=1.7, label='2" / 4" plates'))
    ax_d.add_patch(Circle((0.0, 0.0), 5.08, fill=False, ec="#16a34a", ls="--", lw=1.5))
    ax_d.plot(bd_x, bd_y, marker="*", ms=13, color="#facc15", mec="k", mew=1.0, ls="none", zorder=6,
              label=f"Shot {shot} B-dot ({bd_x:+.0f}, {bd_y:+.0f})".replace("+0", "0").replace("-0", "0"))
    ax_d.plot(lp_x, lp_y, marker="D", ms=7.5, color="#38bdf8", mec="k", mew=1.0, ls="none", zorder=6,
              label=f"Shot {shot} LP ({lp_x:+.0f}, {lp_y:+.0f})".replace("+0", "0").replace("-0", "0"))

    ax_d.set_aspect("equal")
    ax_d.set_xlim(float(xs[0]), float(xs[-1]))
    ax_d.set_ylim(float(ys[0]), float(ys[-1]))
    ax_d.set_xlabel(r"$x$ [cm]")
    ax_d.set_ylabel(r"$y$ [cm]")
    note_lines = [f"$|\\delta B_z| / |\\delta B_\\perp| = {meta['bz_over_bperp']:.2f}$"]
    if np.isfinite(db_over_b0) and db_over_b0 > 0.0:
        exp = int(np.floor(np.log10(db_over_b0)))
        mant = db_over_b0 / (10.0 ** exp)
        note_lines.append(rf"$\delta B_\perp / B_0 = {mant:.1f}\times 10^{{{exp}}}$")
    if np.isfinite(beta_rat):
        note_lines.append(f"$\\beta_e / (m_e/m_i) = {beta_rat:.1f}$")
    ax_d.text(
        0.04, 0.04, "\n".join(note_lines),
        transform=ax_d.transAxes, fontsize=9.8, va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.86),
    )
    ax_d.legend(loc="upper left", frameon=True, framealpha=0.90)
    ax_d.set_title(
        rf"(d) 2-D $J_\parallel$ & $\mathbf{{B}}_\perp$ (Run {run_num}, $f_0 = {meta['f0_hz']*1e-3:.1f}$ kHz)",
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.065, right=0.945, top=0.94, bottom=0.08)
    save_name = "fig1_em_instability" if str(run_num) == "26" else f"fig1_em_instability_run{run_num}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, save_name), show=show)


def plot_sheath_closure(
    run_num="02",
    shot=100,
    *,
    data_dir=None,
    ifn=None,
    modes_npz=None,
    t_window_ms=(-0.5, 5.5),
    zoom_window_ms=(2.8, 3.8),
    shunt_a_per_v=100.0,
    downsample_q=25,
    save_fig=True,
    show=False,
):
    """2-panel Port-35 conducting-sheath current closure figure (default: Run 02).

    * **(a)** Full bias-pulse ``I_2"`` and ``I_4"`` currents [A] showing quiet
      pre-onset DC current erupting into ~4 kHz oscillations at mode onset.
    * **(b)** Zoomed ``zoom_window_ms`` AC current traces (``dI_2"``, ``dI_4"``)
      showing equal-and-opposite 180-deg return current between the ``2"`` disc
      and ``4"`` ring, annotated with ``gamma^2`` and ``dphi`` from
      ``modes_dispersion_jun2026.npz``.
    """
    import Jun2026_modes as jmd

    ddir = data_dir or jmd.DATA_DIR
    hdf5_path = ifn or _run_hdf5(ddir, run_num)
    modes_path = modes_npz or os.path.join(ddir, jmd.OUT_NAME)

    run = open_lapd(hdf5_path)
    tb_ms = np.asarray(run.time_array(scope_name="biasscope"), dtype=float) * 1e3 - jmd.BIASSCOPE_OFFSET_MS
    st_i2, _ = run.channel("C1", scope_name="biasscope", shots=[int(shot)])
    st_i4, _ = run.channel("C2", scope_name="biasscope", shots=[int(shot)])

    t_lo, t_hi = float(t_window_ms[0]), float(t_window_ms[1])
    z_lo, z_hi = float(zoom_window_ms[0]), float(zoom_window_ms[1])
    m_full = (tb_ms >= t_lo) & (tb_ms <= t_hi)
    t_raw = tb_ms[m_full]
    i2_raw = np.asarray(st_i2[0][m_full], dtype=float) * float(shunt_a_per_v)
    i4_raw = np.asarray(st_i4[0][m_full], dtype=float) * float(shunt_a_per_v)

    # Zero tiny pre-bias DC offset using [-0.5, -0.05] ms (bias off -> 0 A)
    m_pre = (t_raw >= t_lo) & (t_raw <= -0.05)
    if np.any(m_pre):
        i2_raw = i2_raw - float(np.mean(i2_raw[m_pre]))
        i4_raw = i4_raw - float(np.mean(i4_raw[m_pre]))

    q = max(int(downsample_q), 1)
    t_ms, i2 = downsample_blockmean(t_raw, i2_raw, q)
    _, i4 = downsample_blockmean(t_raw, i4_raw, q)

    m_z = (t_ms >= z_lo) & (t_ms <= z_hi)
    di2 = i2[m_z] - float(np.mean(i2[m_z]))
    di4 = i4[m_z] - float(np.mean(i4[m_z]))

    g2_24 = dphi_24 = np.nan
    if os.path.exists(modes_path):
        with np.load(modes_path, allow_pickle=True) as md:
            k_g2 = f"bias__{run_num}__g2_I2_I4"
            k_ph = f"bias__{run_num}__dphi_I4_minus_I2_deg"
            if k_g2 in md.files:
                g2_24 = float(md[k_g2])
            if k_ph in md.files:
                dphi_24 = abs(float(md[k_ph]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.2),
                                   gridspec_kw={"height_ratios": [1.15, 1.0]})

    # (a) Full bias pulse
    ax1.axvspan(z_lo, z_hi, color="0.90", alpha=0.65, zorder=0)
    ax1.plot(t_ms, i2, "-", color="#7c3aed", lw=1.6, label=r'Port-35 2" disc ($I_{2^{\prime\prime}}$)')
    ax1.plot(t_ms, i4, "-", color="#ea580c", lw=1.6, label=r'Port-35 4" ring ($I_{4^{\prime\prime}}$)')
    ax1.set_xlim(t_lo, t_hi)
    ax1.set_ylim(-2.0, max(float(np.max(i2)), float(np.max(i4))) * 1.15)
    ax1.set_xlabel("Time relative to bias turn-on [ms]")
    ax1.set_ylabel("End-plate current $I$ [A]")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="lower center", frameon=True, framealpha=0.92)
    ax1.set_title(
        f"(a) Port-35 End-Plate Currents Across Full Bias Pulse (Run {run_num}, shot {shot})",
        fontweight="bold",
    )

    # (b) Zoomed AC closure window
    ax2.axhline(0.0, color="0.45", ls=":", lw=1.0)
    ax2.plot(t_ms[m_z], di2, "-", color="#7c3aed", lw=2.0, label=r'$\delta I_{2^{\prime\prime}}$')
    ax2.plot(t_ms[m_z], di4, "--", color="#ea580c", lw=2.0, label=r'$\delta I_{4^{\prime\prime}}$')
    ax2.set_xlim(z_lo, z_hi)
    ymax = max(float(np.max(np.abs(di2))), float(np.max(np.abs(di4)))) * 1.28
    ax2.set_ylim(-ymax, ymax)
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel(r"Oscillating current $\delta I$ [A]")
    ax2.grid(alpha=0.22)
    if np.isfinite(g2_24) and np.isfinite(dphi_24):
        ax2.text(
            0.03, 0.06,
            f"$\\gamma^2 = {g2_24:.2f},\\; \\Delta\\phi = {dphi_24:.0f}^\\circ$",
            transform=ax2.transAxes, fontsize=10.0, va="bottom",
        )
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax2.set_title(
        f"(b) Zoomed Shaded Window ({z_lo:.1f}–{z_hi:.1f} ms): $180^\\circ$ Out-of-Phase Closure",
        fontweight="bold",
    )

    fig.tight_layout(pad=1.2)
    save_name = "fig2_sheath_closure" if str(run_num) == "02" else f"fig2_sheath_closure_run{run_num}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, save_name), show=show)


def animate_mode_propagation(
    run_num="26",
    *,
    matched_puff_run="03",
    f_exb_khz=None,
    seamless_loop=True,
    data_dir=None,
    bdot_npz=None,
    modes_npz=None,
    n_frames=60,
    fps=15,
    dpi=100,
    r_orbit_cm=3.2,
    save_fig=True,
    preview_png=None,
):
    """Side-by-side GIF animating 2-D ``J_par`` & ``B_perp`` in lab vs. plasma frame.

    Reads ``bdot_2d_jun2026.npz`` (and ``modes_dispersion_jun2026.npz`` for the
    matched ``E x B`` Doppler frequency when ``f_exb_khz is None``) with zero
    raw HDF5 reads:

    * **Left (Lab frame):** measured ``m = -1`` dipole rotates CW at ``f_lab``
      while an ``E x B`` plasma fluid marker rotates CW faster at ``f_ExB``.
    * **Right (Plasma rest frame, co-rotating with ``E x B``):** fluid marker is
      stationary and the dipole rotates CCW at ``f_ExB - f_lab``, along the
      electron ``grad(Te)`` drift and opposite to ``grad(ne)`` drift.

    Returns the saved GIF path (or ``None`` if ``save_fig`` is falsey).
    """
    from matplotlib.patches import Circle, FancyArrowPatch
    from PIL import Image
    from scipy.ndimage import map_coordinates
    import Jun2026_bdot_2d as jbd
    import Jun2026_modes as jmd

    ddir = data_dir or jbd.DATA_DIR
    bdot_path = bdot_npz or os.path.join(ddir, jbd.OUT_NAME)
    modes_path = modes_npz or os.path.join(ddir, jmd.OUT_NAME)

    xs, ys, Bx, By, Jz, meta = _load_bdot_2d_snapshot(bdot_path, run_num)
    cx, cy = meta["center_x_cm"], meta["center_y_cm"]
    f_lab_khz = meta["f0_hz"] * 1e-3

    if f_exb_khz is None:
        with np.load(modes_path, allow_pickle=True) as md:
            f_exb_meas = float(md[f"disp__{matched_puff_run}__f_doppler_exb_hz"]) * 1e-3
    else:
        f_exb_meas = float(f_exb_khz)

    if seamless_loop and f_exb_meas > f_lab_khz:
        # Snap to nearest half-integer harmonic (e.g. 3/2) for an exact loop:
        # 2 lab turns = 3 E x B turns = 1 full CCW plasma-frame turn (0.7% shift on Run 26).
        half_mult = max(int(round(2.0 * f_exb_meas / f_lab_khz)), 3)
        f_exb_eff = 0.5 * half_mult * f_lab_khz
        t_loop_ms = 2.0 / f_lab_khz
    else:
        f_exb_eff = f_exb_meas
        t_loop_ms = 1.0 / max(abs(f_exb_eff - f_lab_khz), 0.1)
    f_pl_khz = f_exb_eff - f_lab_khz

    ph0 = np.angle(Jz[np.unravel_index(int(np.nanargmax(np.abs(Jz))), Jz.shape)])
    Jz = Jz * np.exp(-1j * ph0)
    Bx = Bx * np.exp(-1j * ph0)
    By = By * np.exp(-1j * ph0)

    is_cal = bool(meta["bdot_calibrated"])
    j_max = float(np.nanmax(np.abs(Jz)))
    if is_cal:
        j_scale, j_unit = (1e-3, r"kA/m$^2$") if j_max >= 1e3 else (1.0, r"A/m$^2$")
        Jz_n = Jz * j_scale
        j_vlim = j_max * j_scale
        cb_label = rf"$\mathrm{{Re}}[J_\parallel]$ [{j_unit}]"
    else:
        Jz_n = Jz / j_max
        j_vlim = 1.0
        cb_label = r"$\mathrm{Re}[J_\parallel] / J_{\max}$"
    b_scale = float(np.nanpercentile(np.hypot(np.abs(Bx), np.abs(By)), 95))
    Bx_n = Bx / b_scale
    By_n = By / b_scale

    x0, dx = float(xs[0]), float(xs[1] - xs[0])
    y0, dy = float(ys[0]), float(ys[1] - ys[0])
    xf = np.linspace(float(xs[0]), float(xs[-1]), 129)
    yf = np.linspace(float(ys[0]), float(ys[-1]), 129)
    XF, YF = np.meshgrid(xf, yf)
    XQ, YQ = np.meshgrid(xs, ys)

    def _sample_complex(arr, x_pts, y_pts):
        iy = (y_pts - y0) / dy
        ix = (x_pts - x0) / dx
        re = map_coordinates(np.real(arr), [iy, ix], order=1, mode="constant", cval=0.0)
        im = map_coordinates(np.imag(arr), [iy, ix], order=1, mode="constant", cval=0.0)
        return re + 1j * im

    Jz_lab_f = _sample_complex(Jz_n, XF, YF)

    def _eval_frame(t_ms):
        wt = 2.0 * np.pi * f_lab_khz * t_ms
        alpha = 2.0 * np.pi * f_exb_eff * t_ms
        phase = np.exp(-1j * wt)

        jz_l = np.real(Jz_lab_f * phase)
        bx_l = np.real(Bx_n * phase)
        by_l = np.real(By_n * phase)

        ca, sa = np.cos(alpha), np.sin(alpha)
        dx_f, dy_f = XF - cx, YF - cy
        xl_f = cx + dx_f * ca + dy_f * sa
        yl_f = cy - dx_f * sa + dy_f * ca
        jz_p = np.real(_sample_complex(Jz_n, xl_f, yl_f) * phase)

        dx_q, dy_q = XQ - cx, YQ - cy
        xl_q = cx + dx_q * ca + dy_q * sa
        yl_q = cy - dx_q * sa + dy_q * ca
        bx_lq = np.real(_sample_complex(Bx_n, xl_q, yl_q) * phase)
        by_lq = np.real(_sample_complex(By_n, xl_q, yl_q) * phase)
        bx_p = bx_lq * ca - by_lq * sa
        by_p = bx_lq * sa + by_lq * ca
        return jz_l, bx_l, by_l, jz_p, bx_p, by_p, np.pi / 2.0 - alpha

    def _add_curved_arrow(ax, r, ang1_deg, ang2_deg, color, label, rad, text_ang_deg, r_text):
        a1, a2 = np.deg2rad(ang1_deg), np.deg2rad(ang2_deg)
        arr = FancyArrowPatch(
            (cx + r * np.cos(a1), cy + r * np.sin(a1)),
            (cx + r * np.cos(a2), cy + r * np.sin(a2)),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=6.5,head_width=4.0",
            color=color, lw=2.3, zorder=7,
        )
        ax.add_patch(arr)
        at = np.deg2rad(text_ang_deg)
        ax.text(
            cx + r_text * np.cos(at), cy + r_text * np.sin(at),
            label, color=color, fontsize=9.2, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
            zorder=8,
        )

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 5.5), dpi=int(dpi))
    fig.subplots_adjust(left=0.06, right=0.89, top=0.85, bottom=0.11, wspace=0.16)

    jz_l0, bx_l0, by_l0, jz_p0, bx_p0, by_p0, th0 = _eval_frame(0.0)
    ext = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    im_l = ax_l.imshow(jz_l0, extent=ext, origin="lower", cmap="RdBu_r", vmin=-j_vlim, vmax=j_vlim)
    qv_l = ax_l.quiver(XQ, YQ, bx_l0, by_l0, color="k", scale=14.0, width=0.0045, alpha=0.78)
    im_r = ax_r.imshow(jz_p0, extent=ext, origin="lower", cmap="RdBu_r", vmin=-j_vlim, vmax=j_vlim)
    qv_r = ax_r.quiver(XQ, YQ, bx_p0, by_p0, color="k", scale=14.0, width=0.0045, alpha=0.78)

    for ax in (ax_l, ax_r):
        ax.add_patch(Circle((cx, cy), float(r_orbit_cm), fill=False, ec="#15803d",
                            ls="--", lw=1.1, alpha=0.65, zorder=4))
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel(r"$x$ [cm]")
    ax_l.set_ylabel(r"$y$ [cm]")
    ax_r.set_yticklabels([])

    dot_l, = ax_l.plot(cx + r_orbit_cm * np.cos(th0), cy + r_orbit_cm * np.sin(th0),
                       "o", ms=9.5, color="#22c55e", mec="k", mew=1.2, zorder=9,
                       label=rf"$E\times B$ plasma ({f_exb_meas:.1f} kHz CW)")
    ax_r.plot(cx, cy + r_orbit_cm, "o", ms=9.5, color="#22c55e", mec="k", mew=1.2, zorder=9,
              label=r"$E\times B$ plasma (fixed)")

    _add_curved_arrow(ax_l, 5.3, 165, 110, "0.15", f"Mode ({f_lab_khz:.1f} kHz CW)", -0.28, 140, 6.9)
    _add_curved_arrow(ax_r, 5.3, 110, 165, "#dc2626", r"$\nabla T_e$ drift (CCW)", 0.28, 142, 6.9)
    _add_curved_arrow(ax_r, 5.3, 250, 195, "#1d4ed8", r"$\nabla n_e$ drift (CW)", -0.28, 222, 6.9)

    ax_l.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8.6)
    ax_r.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8.6)
    ax_l.set_title("Lab Frame\n(Mode lags behind CW $E\\times B$ flow)", fontweight="bold")
    ax_r.set_title(
        f"Plasma Rest Frame (co-rotating with $E\\times B$)\n"
        f"(Mode moves CCW at {abs(f_pl_khz):.1f} kHz — with $\\nabla T_e$)",
        fontweight="bold",
    )

    cax = fig.add_axes([0.905, 0.14, 0.018, 0.68])
    cb = fig.colorbar(im_r, cax=cax)
    cb.set_label(cb_label)
    time_txt = fig.suptitle(f"Run {run_num} ({meta['b0_g']:.0f} G)  —  t = 0.00 ms",
                            fontsize=12.0, fontweight="bold", y=0.975)

    n_fr = int(n_frames)
    frames = []
    preview_idx = {0, n_fr // 4, n_fr // 2}
    preview_imgs = []
    for k in range(n_fr):
        t_ms = t_loop_ms * (k / n_fr)
        jz_l, bx_l, by_l, jz_p, bx_p, by_p, th_fl = _eval_frame(t_ms)
        im_l.set_data(jz_l)
        qv_l.set_UVC(bx_l, by_l)
        im_r.set_data(jz_p)
        qv_r.set_UVC(bx_p, by_p)
        dot_l.set_data([cx + r_orbit_cm * np.cos(th_fl)], [cy + r_orbit_cm * np.sin(th_fl)])
        time_txt.set_text(f"Run {run_num} ({meta['b0_g']:.0f} G)  —  t = {t_ms:.2f} ms")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba[:, :, :3])
        frames.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        if preview_png and k in preview_idx:
            preview_imgs.append(img.copy())
    plt.close(fig)

    save_name = "fig3_mode_propagation" if str(run_num) == "26" else f"fig3_mode_propagation_run{run_num}"
    out_gif = plot_utils.fig_path(save_name, FIG_SUBDIR, ext=".gif") if save_fig is True else (save_fig or None)
    if out_gif:
        os.makedirs(os.path.dirname(os.path.abspath(out_gif)), exist_ok=True)
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=max(int(round(1000.0 / float(fps))), 20),
            loop=0,
            optimize=True,
        )
        print(f"Animation saved to: {out_gif} ({n_fr} frames)")

    if preview_png and preview_imgs:
        w, h = preview_imgs[0].size
        strip = Image.new("RGB", (w, h * len(preview_imgs)))
        for i, im in enumerate(preview_imgs):
            strip.paste(im, (0, i * h))
        strip.save(preview_png)

    return out_gif


if __name__ == '__main__':


    iv_ifn = r"D:\data\LAPD\jun2026-jia\06-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"
    # isat_ifn = r"D:\data\LAPD\jun2026-jia\07-He-800G-bias40V-Isat-p29-plane_2026-06-10.hdf5"
    plot_iv_line_run(iv_ifn, save_fig=True, show=False)
