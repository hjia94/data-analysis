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

def finalize_figure(fig, save_fig=None, show=False, compact_axes=None):
    """Save and/or show a finished figure, then release it.

    ``save_fig`` -- path to write the PNG to; ``None`` skips saving.
    ``show``     -- call ``plt.show()`` (the interactive window owns the figure
                    until closed); otherwise the figure is closed immediately
                    (headless/batch saving).
    ``compact_axes`` -- optional group of shared-x axes to pack tight *after*
                    ``tight_layout`` (which would otherwise re-space them); see
                    :func:`_pack_shared_x`.
    """
    fig.tight_layout()
    if compact_axes is not None:
        _pack_shared_x(compact_axes)
    if save_fig is not None:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_fig}")
    if show:
        plt.show()
    else:
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

def _pack_shared_x(axes, gap=0.012):
    """Stack ``axes`` so they abut vertically with only ``gap`` between them.

    For a group of panels that share one x-axis the default ``subplots`` spacing
    leaves wasted blank rows between them.  This repositions the panels to fill
    their *current* combined top->bottom extent, split into equal heights with a
    thin ``gap`` (figure fraction) between neighbours.  Only the passed ``axes``
    move, so any other panels on the same figure (e.g. a trailing Isat/FFT block)
    keep their original positions and stay visually separated.
    """
    boxes = [ax.get_position() for ax in axes]
    top = max(b.y1 for b in boxes)
    bottom = min(b.y0 for b in boxes)
    left = boxes[0].x0
    width = boxes[0].width
    n = len(axes)
    h = (top - bottom - gap * (n - 1)) / n
    for i, ax in enumerate(axes):
        y0 = top - (i + 1) * h - i * gap
        ax.set_position([left, y0, width, h])


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
                 show=True, title=None, isat=None, fft=None,
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
    ``show`` -- call ``plt.show()`` (set False for headless/batch saving only).
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
    _draw_isat_panel(axs[4], isat, marked_times)
    _draw_fft_panel(axs[5], fft, fft_fmax_khz,
                    f"Isat averaged FFT (all shots), '{ISAT_SCOPE}'/{ISAT_CHAN}")

    # `show` is currently a no-op for this section (the batch drivers save
    # headlessly); kept in the signature for symmetry with the combined figure.
    finalize_figure(fig, save_fig=save_fig, compact_axes=axs[:4])


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


def _draw_isat_panel(ax, isat, marked_times):
    """Draw the Isat reference panel (5th panel of the IV line-scan figure).

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
        _scatter_iv_times_on_isat(ax, t_ms, I, marked_times)
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

    # Reference Isat trace for the 5th panel -- read from the raw HDF5 (it's not
    # in the saved .npz).  None when there's no run file or the channel is absent,
    # in which case the panel is drawn blank.
    isat = _read_isat_trace(run, npos, nshot) if run is not None else None

    # All-shot-averaged Isat FFT for the last panel (same channel as the trace).
    # None when there's no run file or the channel is absent -> blank panel.
    fft = _read_isat_fft(ifn) if ifn is not None else None

    name = f"{run_num}{jiv._tip_tag(load_tip)}-line"
    save_path = fig_path(name) if save_fig is True else (save_fig or None)
    plot_iv_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, ndx,
                 save_fig=save_path, title=title, isat=isat, fft=fft)


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
#  Reuses the IV loaders (jiv.load_data / load_sweep_data), the Isat reader (jis)
#  and run_title -- no read/FFT logic is re-implemented here.
# =========================================================================== #

def puff_title(ifn, run=None):
    """The gas-puff span of a run's description (``"Puff voltage 75V for 30ms"``).

    Reuses :func:`run_title` (which pulls the puff span via ``_PUFF_RE``) and
    strips its leading ``"<run_num>-"`` so the result is just the puff
    voltage/time setting -- used as the combined figure's title.  Returns
    ``None`` when no description / puff line is available.
    """
    title = run_title(ifn, run_num_of(ifn), run=run)
    if not title:
        return None
    m = _PUFF_RE.search(title)
    return m.group(0) if m else title


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
    ``iv_tip`` selects the tip whose .npz to load (e.g. ``"L"``).  ``fft_npz``
    defaults to ``<isat dir>/Jun2026_Isat.OUT_NPZ``; the Isat run must be one of
    the runs in that batch npz.  The figure title is the gas-puff
    voltage/time from the IV run's description.  ``save_fig`` False -> don't save;
    ``show`` True pops the interactive window (set False for headless saving).
    """
    data_dir = os.path.dirname(iv_ifn)
    run_num = run_num_of(iv_ifn)
    isat_run_num = run_num_of(isat_ifn)
    tndx_list = tndx_list if tndx_list is not None else [-7, -5, -3, -1]

    # Panels 1-3: IV arrays + axes from the IV run's saved .npz.
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = jiv.load_data(data_dir, run_num, tip=iv_tip)
    _, _, _, xpos, _, _, _ = jiv.load_sweep_data(data_dir, run_num, tip=iv_tip)

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

    # Panel 5: Isat raw, shot-averaged, 0..isat_tmax_ms; y starts at 0.  The IV
    # sweep instants are scattered on the line, coloured to match the IV panels.
    t_isat_ms = t_isat * 1e3
    axs[4].plot(t_isat_ms, I_avg, "k", linewidth=0.8, zorder=1)
    _scatter_iv_times_on_isat(axs[4], t_isat_ms, I_avg, marked_times)
    axs[4].set_xlim(0, isat_tmax_ms)
    axs[4].set_ylim(0,1)
    axs[4].set_ylabel("Isat [a.u.]", fontsize=11)
    axs[4].set_xlabel("t [ms]", fontsize=11)
    axs[4].grid(True, alpha=0.3)

    # Panel 6: all-shot-averaged FFT for the Isat run, < fft_fmax_khz.
    _draw_fft_panel(axs[5], fft, fft_fmax_khz,
                    f"Run {isat_run_num} Isat averaged FFT (all shots), "
                    f"scope '{isat_scope}' {isat_chan}",
                    chan_label=f"'{isat_scope}'/{isat_chan}")

    name = f"{run_num}{jiv._tip_tag(iv_tip)}-{isat_run_num}-combined"
    save_path = fig_path(name) if save_fig is True else (save_fig or None)
    finalize_figure(fig, save_fig=save_path, show=show, compact_axes=axs[:4])


# =========================================================================== #
#  Figure: cross-correlation (coherence / cross-phase / lag) for a channel pair
#
#  Jun2026_xcorr.py does the reading + DSP and returns plain dicts/arrays; the
#  figures live here, like every other Jun-2026 figure (Jun2026_Isat.py is
#  likewise plot-free).  Three drivers share one panel drawer:
#    plot_xcorr_per_shot(per_shot)  -- overlay every shot (from jxc.xcorr_per_shot)
#    plot_xcorr_averaged(avg)       -- the ensemble result (from jxc.xcorr_averaged)
#    plot_xcorr_run(ifn)            -- reload a run's ensemble result from batch npz
#  The lag (cross-correlation) panel is optional: the per-shot/averaged figures
#  have it; the batch npz doesn't store it, so plot_xcorr_run omits it.
# =========================================================================== #

def mask_phase_by_coherence(phase_rad, gamma2, gamma2_floor=0.5):
    """Cross-phase in **degrees**, NaN where ``gamma2 < gamma2_floor``.

    Phase is only trustworthy where the two channels are coherent, so incoherent
    bands are blanked (NaN) before plotting.
    """
    return np.where(gamma2 >= gamma2_floor, np.degrees(phase_rad), np.nan)


def _draw_xcorr_panels(axs, freq_khz, gamma2, phase_deg, lags_us=None,
                       xcorr=None, gamma2_floor=0.5, mark_peak=False, **line_kw):
    """Draw one trace across the coherence / cross-phase (/ lag) panels of ``axs``.

    ``axs`` is 2 axes (coherence, cross-phase) or 3 (plus the time-lag panel).
    Call once per shot to overlay faint lines, or once for a bold ensemble line;
    ``**line_kw`` (``color``/``alpha``/``lw``) styles all panels together.  The
    phase panel is masked to ``gamma2 >= gamma2_floor``.  The lag panel is drawn
    only when ``axs`` has a third axis and ``lags_us``/``xcorr`` are given;
    ``mark_peak`` then marks the cross-correlation peak.
    """
    axs[0].plot(freq_khz, gamma2, **line_kw)
    axs[1].plot(freq_khz, np.where(gamma2 >= gamma2_floor, phase_deg, np.nan),
                **line_kw)
    if len(axs) > 2 and lags_us is not None and xcorr is not None:
        axs[2].plot(lags_us, xcorr, **line_kw)
        if mark_peak:
            k = int(np.argmax(xcorr))
            axs[2].scatter(lags_us[k], xcorr[k], s=60, zorder=3,
                           edgecolors="k", linewidths=0.5,
                           color=line_kw.get("color", "C0"))


def _finish_xcorr_axes(axs, fmax_khz, title):
    """Label / limit the shared xcorr axes (2 or 3) and title the top panel."""
    axs[0].set_ylabel(r"coherence $\gamma^2$")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, fmax_khz)
    axs[0].grid(True, alpha=0.3)
    if title:
        axs[0].set_title(title, fontsize=12, fontweight="bold")

    axs[1].set_ylabel(r"cross-phase $\Delta\phi$ [deg]")
    axs[1].set_ylim(-180, 180)
    axs[1].set_xlim(0, fmax_khz)
    axs[1].set_xlabel("frequency [kHz]")
    axs[1].grid(True, alpha=0.3)

    if len(axs) > 2:
        axs[2].set_ylabel("cross-correlation")
        axs[2].set_xlabel(r"lag [$\mu$s]")
        axs[2].grid(True, alpha=0.3)


def plot_xcorr_per_shot(per_shot, title=None, fmax_khz=80.0, gamma2_floor=0.5,
                        save_fig=None, show=True):
    """Overlay every shot's 3-panel correlation (from ``jxc.xcorr_per_shot``).

    Each shot is a faint line so the shot-to-shot spread is visible.  ``save_fig``
    True routes through :func:`fig_path`; a string is an explicit path; falsey
    skips saving (same convention as the rest of Jun-2026).
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    freq_khz = per_shot["freq"] * 1e-3
    lags_us = per_shot["lags"] * 1e6
    nshot = per_shot["gamma2"].shape[0]
    for i in range(nshot):
        _draw_xcorr_panels(axs, freq_khz, per_shot["gamma2"][i],
                           np.degrees(per_shot["phase"][i]), lags_us,
                           per_shot["xcorr"][i], gamma2_floor=gamma2_floor,
                           color="C0", alpha=0.25, lw=0.8)
    _finish_xcorr_axes(axs, fmax_khz,
                       title or f"Per-shot cross-correlation ({nshot} shots)")
    save_path = fig_path(save_fig) if save_fig is True else (save_fig or None)
    finalize_figure(fig, save_fig=save_path, show=show)


def plot_xcorr_averaged(avg, title=None, fmax_khz=80.0, gamma2_floor=0.5,
                        save_fig=None, show=True):
    """Draw the ensemble-averaged 3-panel correlation (from ``jxc.xcorr_averaged``).

    One bold line per panel; the cross-correlation peak is marked.  ``save_fig``
    follows the same convention as :func:`plot_xcorr_per_shot`.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    freq_khz = avg["freq"] * 1e-3
    lags_us = avg["lags"] * 1e6
    _draw_xcorr_panels(axs, freq_khz, avg["gamma2"], np.degrees(avg["phase"]),
                       lags_us, avg["xcorr"], gamma2_floor=gamma2_floor,
                       mark_peak=True, color="k", lw=1.2)
    n = avg.get("n_used", "?")
    _finish_xcorr_axes(axs, fmax_khz,
                       title or f"Ensemble-averaged cross-correlation ({n} shots)")
    save_path = fig_path(save_fig) if save_fig is True else (save_fig or None)
    finalize_figure(fig, save_fig=save_path, show=show)


def plot_xcorr_run(ifn, ch_a=jxc.CH_A, ch_b=jxc.CH_B, npz_path=None,
                   save_fig=True, show=False, fmax_khz=80.0, gamma2_floor=0.5):
    """Draw a channel pair's ensemble coherence + cross-phase from the run's npz.

    Loads the pair's ``<pair>__gamma2`` / ``<pair>__phase`` (keyed by
    :func:`Jun2026_xcorr._pair_key`) against the shared ``freq`` axis from the
    run's co-located npz (default :func:`Jun2026_xcorr.xcorr_npz_path`) -- no HDF5
    read, no recomputation.  Run ``Jun2026_xcorr.batch_xcorr`` first.  The npz
    doesn't store the lag trace, so this figure is the two frequency-domain panels
    only (reusing :func:`_draw_xcorr_panels` with no lag axis).  ``save_fig`` True
    routes through :func:`fig_path`.
    """
    if npz_path is None:
        npz_path = jxc.xcorr_npz_path(ifn)
    key = jxc._pair_key(ch_a, ch_b)
    try:
        with np.load(npz_path) as d:
            freq = d["freq"]
            gamma2 = d[f"{key}__gamma2"]
            phase = d[f"{key}__phase"]
    except (OSError, KeyError) as e:
        print(f"  (xcorr: no saved entry for pair '{key}' in {npz_path} -- {e})")
        return

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    _draw_xcorr_panels(axs, freq * 1e-3, gamma2, np.degrees(phase),
                       gamma2_floor=gamma2_floor, color="k", lw=1.0)
    title = run_title(ifn, run_num_of(ifn)) or f"{run_num_of(ifn)} xcorr"
    _finish_xcorr_axes(axs, fmax_khz, f"{title}  —  {ch_a[0]}/{ch_a[1]} vs {ch_b[0]}/{ch_b[1]}")
    name = f"{run_num_of(ifn)}-xcorr-{ch_a[0]}{ch_a[1]}-{ch_b[0]}{ch_b[1]}"
    save_path = fig_path(name) if save_fig is True else (save_fig or None)
    finalize_figure(fig, save_fig=save_path, show=show)


# =========================================================================== #
#  Add further Jun-2026 figure sections below, each following the IV pattern:
#    plot_<fig>(...)            -- draw one figure, ending in finalize_figure()
#    plot_<fig>_run(ifn, ...)   -- load the saved .npz and call plot_<fig>
# =========================================================================== #


if __name__ == '__main__':

    iv_ifn=r"D:\data\LAPD\jun2026-jia\25-He-800G-bias40V-LP-p29-line_2026-06-12.hdf5"
    isat_ifn=r"D:\data\LAPD\jun2026-jia\05-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"
    
    # plot_iv_line_run(iv_ifn, tndx_list=[-3,-2,-1], save_fig=True)

    # Combined IV (run 25, tip R) + Isat raw/FFT (run 05) on one figure.
    plot_iv_isat_combined(iv_ifn, isat_ifn, iv_tip="R", tndx_list=[-16,-14,-12,-10,-8], show=False, save_fig=True)
