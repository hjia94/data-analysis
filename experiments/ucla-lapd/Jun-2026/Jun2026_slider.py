"""Jun-2026 slider *adapters*: saved analysis npz -> a scrubbable HTML page.

The third piece of the Jun-2026 split. ``Jun2026_xcorr`` (and friends) read raw
HDF5 and save arrays; ``Jun2026_plot`` draws static publication figures; this
module builds :mod:`data_analysis.viz.slider_html` bundles from those same saved
arrays and renders a self-contained interactive page.

Every adapter follows the same five steps:

1. load the saved arrays for one product,
2. slice the scan axis to the range worth scrubbing,
3. grid the frames onto the spatial layout the *positions* imply
   (:func:`data_analysis.viz.plot_utils.grid_frames`),
4. build the bundle dict -- typed axis, geometry, one entry per field,
5. hand it to :func:`data_analysis.viz.slider_html.write_slider_html`.

One page or many
--------------------------------------------------------------------------
:func:`emit_iv_line_slider_all` puts **every IV run on one page**, the runs
behind the dropdown and both probe tips overlaid in each frame.
:func:`emit_xcorr_slider` renders one channel pair per page.
:func:`emit_isat_fft_slider` puts **one run's Isat tips on one page**, the
slider scrubbing probe position and each tip a trace -- the schema's
non-spatial case, where the panel's x-axis is frequency rather than position.
:func:`emit_isat_fft_slider_all` puts **every such run on one page**, the runs
behind the dropdown and the tips still overlaid: the runs that repeat one probe
line while a setting is walked (00-06, gas puff 23->45 ms) are read by switching
that dropdown at a held position.
:func:`emit_xcorr_slider_all` puts **every pair of a run on one page** behind a
dropdown, which is the one to reach for when the question is "which channel pair
shows this?" -- comparing pairs by switching a dropdown at a held frequency,
rather than by tiling browser windows.

Each *_all page shares its per-item preparation with the single-item emitter
(:func:`_xcorr_pair_fields` for a pair, :func:`_isat_run_read` for a run), so a
change to how one is prepared cannot apply to only one of the two pages.

A dropdown of runs and a dropdown of channels are the same schema feature, so
what a page's dropdown *varies* is stated in its ``group_label`` -- the caption
is the only thing on the page that says whether switching it changes the
channel or the run.
"""

import glob
import os
from collections import namedtuple

import numpy as np

from data_analysis.io import open_lapd, parse_gas_puff
from data_analysis.plasma.langmuir import (discover_channels, load_plasma_data,
                                           load_sweep_axes, ne_is_calibrated)
from data_analysis.utils import run_num_of
from data_analysis.viz.plot_utils import fig_path, grid_frames
from data_analysis.viz.slider_html import SCHEMA_VERSION, write_slider_html

import Jun2026_xcorr as jxc
import Jun2026_plot as jpl
import Jun2026_Isat as jis
import Jun2026_IV as jiv


# Default subdirectory under $DATA_ANALYSIS_OUTPUT/figures/
FIG_SUBDIR = jpl.FIG_SUBDIR

# Upper limit of the frequency band rendered to the html page, kHz.
# For correlation and Isat FFT
XCORR_FMAX_KHZ = 20.0


#: Shown on html page when no coherence floor was applied.
_XCORR_WARNING = (
    "No coherence floor applied — cross-phase is random noise wherever γ² is "
    "near zero. Read the phase map only where the coherence map beside it is high.")


def slider_path(name, subdir=FIG_SUBDIR):
    """Centralized ``.html`` location: :func:`data_analysis.viz.plot_utils.fig_path`
    with this campaign's default subdirectory and an ``.html`` extension.
    """
    return fig_path(name, subdir, ext=".html")


def _xcorr_params(freq_khz, fmax_khz, gamma2_floor, stored=None):
    """The provenance ``params`` block shared by both xcorr pages.

    ``stored`` is the npz's own record of the batch settings
    (:func:`Jun2026_xcorr.stored_settings`).
    """
    stored = stored or {}
    window = stored.get("window")
    nperseg = stored.get("nperseg")
    params = {"band": f"0-{fmax_khz:g} kHz ({freq_khz.size} bins)",
              "window": window or f"{jxc.TMIN_MS}-{jxc.TMAX_MS} ms (not recorded)",
              "nperseg": nperseg if nperseg is not None
                         else f"{jxc.NPERSEG} (not recorded)"}
    if gamma2_floor is not None:
        params["phase masked below gamma2"] = gamma2_floor
    return params


def _xcorr_pair_fields(ifn, ch_a, ch_b, npz_path, fmax_khz, gamma2_floor):
    """Prepare one channel pair into slider *fields*: band-limit, gate, grid.
    """
    loaded = jpl._load_xcorr_run(ifn, ch_a, ch_b, npz_path)
    if loaded is None:
        return None
    freq, gamma2, phase, pos_x, pos_y = loaded

    # Band-limit first: everything below is proportional to the number of bins.
    # The DC bin always passes, so test for a scrubbable band (>= 2 bins) rather
    # than a non-empty one -- a one-frame slider has nothing to scrub.
    band = freq <= fmax_khz * 1e3
    if band.sum() < 2:
        raise ValueError(
            f"{fmax_khz} kHz leaves only {band.sum()} frequency bin(s) -- nothing "
            f"to scrub. The axis has df = {(freq[1] - freq[0]) * 1e-3:.3g} kHz, so "
            f"fmax_khz must be at least {(freq[1] - freq[0]) * 1e-3:.3g}.")
    freq_khz = freq[band] * 1e-3
    gamma2 = gamma2[:, band]
    phase_deg = np.degrees(phase[:, band])

    # Optional coherence gate on the phase map only (see emit_xcorr_slider).
    if gamma2_floor is not None:
        phase_deg = np.where(gamma2 < gamma2_floor, np.nan, phase_deg)

    g2_frames, xs, ys = grid_frames(pos_x, pos_y, gamma2)
    ph_frames, _, _ = grid_frames(pos_x, pos_y, phase_deg)

    fields = [
        {"name": "coherence γ²", "unit": "", "frames": g2_frames,
         "cmap": "viridis", "vmin": 0.0, "vmax": 1.0},
        {"name": "cross-phase Δφ", "unit": "deg", "frames": ph_frames,
         "cmap": "twilight", "vmin": -180.0, "vmax": 180.0},
    ]
    return fields, freq_khz, xs, ys


def emit_xcorr_slider(ifn, ch_a=jxc.CH_A, ch_b=jxc.CH_B, npz_path=None,
                      fmax_khz=XCORR_FMAX_KHZ, gamma2_floor=None, out=None):
    """Frequency-slider page for one channel pair of a batched xcorr plane run.

    Reads the pair's per-position ensemble spectra from the run's co-located npz
    (written by :func:`Jun2026_xcorr.batch_xcorr`) and renders **coherence and
    cross-phase side by side**, both scrubbing together over frequency.  The
    static counterpart, :func:`Jun2026_plot.plot_xcorr_plane_run`, collapses the
    whole spectrum to one hand-picked frequency; here every bin in the band is a
    frame, so finding the interesting frequency is a drag rather than a re-run.

    Both maps are shown because cross-phase is only meaningful where the channels
    are actually coherent: ``angle(<Pxy>)`` is defined everywhere but is uniformly
    random where ``gamma2`` is near zero, which reads as spatial structure while
    being noise.  Seeing the coherence map beside the phase map at the *same*
    frequency is how you judge which regions to believe.

    Args:
        ifn (str): The run's raw HDF5 path (only its name/dir are used, to find
            the co-located npz and label the page).
        ch_a, ch_b (tuple): ``(scope, channel)`` pairs identifying the stored
            entry, as passed to :func:`Jun2026_xcorr.batch_xcorr`.
        npz_path (str): Override the co-located npz location.
        fmax_khz (float): Upper edge of the frequency band to ship.
        gamma2_floor (float): Optional coherence threshold; positions below it
            have their **phase** blanked (NaN, drawn grey), matching
            ``plot_xcorr_plane_run``'s masking.  Default ``None`` -- no masking.
            A floor is a *frequency-dependent* judgement (coherence varies
            strongly across the band), so baking one absolute value into every
            frame would mask each frame by a threshold calibrated for a different
            frequency.  Shipping the coherence map instead is what replaces it;
            when a floor *is* set it is recorded in the page's provenance,
            because a masked map that doesn't say it is masked is a trap.
        out (str): Explicit output ``.html`` path; default is
            :func:`slider_path` under the centralized output root.

    Returns the written ``.html`` path, or ``None`` if the pair has no saved
    entry (the reason is printed, so a caller looping over pairs can continue).
    """
    prepared = _xcorr_pair_fields(ifn, ch_a, ch_b, npz_path, fmax_khz,
                                  gamma2_floor)
    if prepared is None:
        return None
    fields, freq_khz, xs, ys = prepared

    run_num = run_num_of(ifn)
    # The same descriptive title the static figure uses (run number + gas-puff
    # setting, the knob that changes run to run), so a page and its PNG are
    # recognizably the same run.
    run_label = jpl.run_title(ifn, run_num) or f"{run_num} xcorr plane"
    pair_label = jxc.pair_label(ch_a, ch_b)
    params = _xcorr_params(freq_khz, fmax_khz, gamma2_floor,
                           jxc.stored_settings(npz_path or jxc.xcorr_npz_path(ifn)))

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": f"{run_label} — {pair_label}",
        "geometry": "plane",
        "axis": {"name": "frequency", "unit": "kHz", "values": freq_khz},
        "x": {"label": "X position", "unit": "cm", "values": xs},
        "y": {"label": "Y position", "unit": "cm", "values": ys},
        "fields": fields,
        "provenance": {"source": os.path.basename(ifn), "params": params,
                       "details": _channel_details(ifn, [ch_a, ch_b])},
        "warning": _XCORR_WARNING if gamma2_floor is None else None,
    }

    name = (f"{run_num}-xcorr-slider-{ch_a[0]}{ch_a[1]}-{ch_b[0]}{ch_b[1]}"
            f"-0to{fmax_khz:g}kHz")
    return write_slider_html(bundle, out or slider_path(name))


def emit_xcorr_slider_all(ifn, npz_path=None, pairs=None,
                          fmax_khz=XCORR_FMAX_KHZ, gamma2_floor=None, out=None):
    """**One** page holding every channel pair of a run, behind a dropdown.

    The whole-run counterpart of :func:`emit_xcorr_slider`.  Every pair shares
    this run's frequency axis and probe plane -- only the measured signal
    differs -- so they belong on one page as selectable channels rather than in
    as many browser tabs as the run has pairs.  The frequency stays put when the
    channel changes, which is what makes "which pair shows structure *here*" a
    question you answer by clicking rather than by re-finding the same bin on
    another page.

    A run's npz accumulates pairs under separate key prefixes, so this asks
    :func:`Jun2026_xcorr.stored_pairs` what is actually stored rather than making
    the caller name each pair.

    Args:
        ifn (str): The run's raw HDF5 path (used to find the co-located npz and
            label the page).
        npz_path (str): Override the co-located npz location.
        pairs (list): Explicit ``[(ch_a, ch_b), ...]`` to include, in dropdown
            order; default is every stored pair, sorted.
        fmax_khz (float): Upper edge of the frequency band to ship.
        gamma2_floor (float): Optional coherence threshold blanking **phase**;
            see :func:`emit_xcorr_slider`.  Applied identically to every pair.
        out (str): Explicit output ``.html`` path.

    Returns the written ``.html`` path, or ``None`` if the run has no stored
    pair to render (the reason is printed).
    """
    if npz_path is None:
        npz_path = jxc.xcorr_npz_path(ifn)
    if pairs is None:
        pairs = jxc.stored_pair_tuples(npz_path)

    groups, shared, rendered = [], None, []
    for ch_a, ch_b in pairs:
        prepared = _xcorr_pair_fields(ifn, ch_a, ch_b, npz_path, fmax_khz,
                                      gamma2_floor)
        if prepared is None:      # missing entry: already reported, skip it
            continue
        fields, *axes = prepared
        label = jxc.pair_label(ch_a, ch_b)
        rendered.append((ch_a, ch_b))

        if shared is None:
            shared, shared_label = axes, label
        elif not all(np.array_equal(a, b) for a, b in zip(axes, shared)):
            raise ValueError(
                f"pair '{label}' has a different frequency axis or probe plane "
                f"than '{shared_label}'; they cannot share one page. Re-run the "
                "xcorr batch so every pair is analysed on the same grid.")

        groups.append({"name": label, "fields": fields})

    if not groups:
        print(f"  (xcorr: no stored pair to render in {npz_path})")
        return None
    freq_khz, xs, ys = shared

    run_num = run_num_of(ifn)
    run_label = jpl.run_title(ifn, run_num) or f"{run_num} xcorr plane"

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": f"{run_label} — {len(groups)} channel pairs",
        "geometry": "plane",
        "axis": {"name": "frequency", "unit": "kHz", "values": freq_khz},
        "x": {"label": "X position", "unit": "cm", "values": xs},
        "y": {"label": "Y position", "unit": "cm", "values": ys},
        "groups": groups,
        "provenance": {"source": os.path.basename(ifn),
                       "params": _xcorr_params(freq_khz, fmax_khz, gamma2_floor,
                                               jxc.stored_settings(npz_path)),
                       # Every channel the dropdown's pairs are built from, each
                       # named once however many pairs it takes part in.
                       "details": _channel_details(
                           ifn, [c for pair in rendered for c in pair])},
        "warning": _XCORR_WARNING if gamma2_floor is None else None,
    }

    name = f"{run_num}-xcorr-slider-all-0to{fmax_khz:g}kHz"
    return write_slider_html(bundle, out or slider_path(name))


def emit_xcorr_sliders(ifn, npz_path=None, **kwargs):
    """Render a **separate** page per stored channel pair; returns their paths.

    Prefer :func:`emit_xcorr_slider_all` -- one page with a channel dropdown.
    This stays for the case where a single pair has to travel on its own (an
    email, a slide), where a page carrying every other pair of the run is more
    than is wanted.
    """
    if npz_path is None:
        npz_path = jxc.xcorr_npz_path(ifn)

    paths = []
    for ch_a, ch_b in jxc.stored_pair_tuples(npz_path):
        path = emit_xcorr_slider(ifn, ch_a, ch_b, npz_path=npz_path, **kwargs)
        if path is not None:
            paths.append(path)
    return paths


# =========================================================================== #
#  IV line scan: runs behind the dropdown, both tips overlaid per frame
# =========================================================================== #

#: Fields drawn per frame, in panel order. ``Te*ne`` is derived rather than
#: stored, matching how ``Jun2026_plot._draw_iv_panels`` computes it. The two
#: lists are deliberately separate: that one carries mathtext labels for
#: matplotlib, this one plain text for a canvas. Keep the quantities in step.
IV_FIELDS = (("Vp", "V"), ("Te", "eV"), ("ne", "cm^-3"), ("Te*ne", "eV cm^-3"))

#: How far two tips' recorded sweep times may differ and still count as the
#: same acquisition [ms]. The tips are timed independently, so their timestamps
#: carry ~1e-4 ms of clock jitter against a sweep spacing of ~0.4 ms; this sits
#: well above the jitter and well below one sweep, so a genuinely offset run
#: still fails. A property of the DAQ clock, not of any one run's spacing.
IV_TIP_JITTER_MS = 0.02

#: Half-width of the rendered probe line [cm]; positions outside |x| > this are
#: dropped before the frames are built, so they leave the colour/y scaling as
#: well as the plot. The scan runs to +-30 cm, but past ~25 cm the probe is in
#: the density skirt where the IV fit is noise-dominated: those points are not
#: the physics of interest and their scatter otherwise sets the scale for every
#: frame. Widen it to see them again -- nothing else depends on the value.
IV_X_LIMIT_CM = 25.0


def _iv_run_group(data_dir, run_num, tips):
    """One run as a slider *group*: per-tip traces of each field, vs position.

    Returns ``(group, xpos, ne_calibrated)``, or ``None`` when no tip of the run
    could be read. Each field carries one trace per tip *this run digitized* --
    one, two, three -- so a frame shows them against one shared y-axis, the
    comparison the static figure could only make by drawing separate lines per
    timestamp. Runs need not agree on their tips: the page's groups carry their
    own trace names.

    ``ne`` is [cm^-3] either way; ``ne_calibrated`` is True only when *every*
    tip of the run was interferometer-scaled, since a frame overlays the tips on
    one axis and an uncalibrated tip's absolute scale rests on ``Aprobe`` and
    the vth convention instead of a measured line density.
    """
    traces_by_field = {name: [] for name, _ in IV_FIELDS}
    ne_calibrated = True
    axis_values, xpos = None, None

    for tip in tips:
        try:
            Vp, Te, ne, *_errs, t_ls = load_plasma_data(data_dir, run_num, tip=tip)
            xs, *_ = load_sweep_axes(data_dir, run_num, tip=tip)
            ne_calibrated &= ne_is_calibrated(data_dir, run_num, tip=tip)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            # One tip short is a smaller page, not a lost run: the other tips
            # are complete measurements and the schema lets this group carry
            # fewer traces than its neighbours.
            print(f"  (IV: run {run_num} tip {tip} unreadable: {exc}; "
                  "skipping the tip)")
            continue

        # Crop data in xarray space, where no plasma exists
        keep = np.abs(np.asarray(xs, float)) <= IV_X_LIMIT_CM
        xs = np.asarray(xs, float)[keep]
        Vp, Te, ne = Vp[keep], Te[keep], ne[keep]

        # Some runs store one more timestamp than they have sweeps.
        n_sweeps = Vp.shape[1]
        times_ms = np.asarray(t_ls, float)[:n_sweeps] * 1e3


        if axis_values is None:
            axis_values, xpos, first_tip = times_ms, np.asarray(xs, float), tip
        else:
            same_time = (times_ms.shape == axis_values.shape
                         and np.allclose(times_ms, axis_values, rtol=0,
                                         atol=IV_TIP_JITTER_MS))
            if not (same_time and np.array_equal(np.asarray(xs, float), xpos)):
                print(f"  (IV: run {run_num} tip {tip} disagrees with tip "
                      f"{first_tip} on the time axis or probe line; skipping "
                      "the tip)")
                continue

        # Saved as (npos, n_sweeps); the schema wants (n_axis, nx). Zipped
        # against IV_FIELDS so the panel order is stated once.
        for (name, _), cube in zip(IV_FIELDS,
                                   (Vp.T, Te.T, ne.T, (Te * ne).T)):
            traces_by_field[name].append(
                {"name": f"tip {tip}" if tip else "tip", "frames": cube})

    if axis_values is None:              # every tip was skipped above
        print(f"  (IV: run {run_num} has no readable tip; skipping the run)")
        return None

    group = {
        "name": "",                      # filled in by the caller (delay time)
        # The first tip's timestamps label the frames; the check above has
        # established the others match to well within a sweep.
        "axis": {"name": "time", "unit": "ms", "values": axis_values},
        # Units are the placeholders from IV_FIELDS; the caller rewrites the ne
        # pair once it knows whether EVERY run calibrated -- the schema requires
        # one field signature across groups, so a per-run unit cannot stand.
        "fields": [{"name": name, "unit": unit, "cmap": "viridis",
                    # No natural physical scale, and the per-frame range varies
                    # by ~10x across a run, so these follow the page's toggle.
                    "vmin": None, "vmax": None,
                    "traces": traces_by_field[name]}
                   for name, unit in IV_FIELDS],
    }
    return group, xpos, ne_calibrated


def emit_iv_line_slider_all(ifns, out=None, name=None):
    """Every IV run on **one** page: runs in the dropdown, tips overlaid.

    Args:
        ifns: The run ``.hdf5`` paths, in dropdown order. Each needs its saved
            ``-plasma-data.npz`` beside it (:func:`Jun2026_IV.process_run`).
        out: Destination ``.html``; defaults under :func:`slider_path`.
        name (str): Output stem, when the default is not wanted.

    Every run's ne is [cm^-3], so calibrated and uncalibrated runs share a page
    and an axis; the provenance line names any run whose absolute scale is not
    interferometer-backed rather than dropping it.

    Runs keep their own time axes: they differ in both sweep count and span, so
    the slider holds the frame *index* across a switch and the readout shows the
    real time for the run now selected.
    """
    groups, xpos, shared_label = [], None, None
    uncalibrated = []              # run labels whose ne is not interferometer-scaled
    for ifn in ifns:
        data_dir = os.path.dirname(ifn)
        run_num = run_num_of(ifn)
        try:
            tips = discover_channels(ifn)
        except FileNotFoundError:
            print(f"  (IV: run {run_num} has no saved sweep data; skipping)")
            continue

        prepared = _iv_run_group(data_dir, run_num, tips)
        if prepared is None:
            continue
        group, xs, run_ne_calibrated = prepared

        # Runs keep their acquisition order here, unlike the Isat page: an IV
        # group's frames are timestamps, and the sort key is discarded.
        group["name"], _ = _puff_run_entry(ifn, run_num)

        # One page carries one probe line for every run. As with the tips, a
        # page that adopted one run's positions and drew another run's data
        # against them would be wrong invisibly.
        if xpos is None:
            xpos, shared_label = xs, group["name"]
        elif not np.array_equal(xs, xpos):
            raise ValueError(
                f"{group['name']} was measured on a different probe line than "
                f"{shared_label}; they cannot share one page's x-axis.")
        groups.append(group)
        if not run_ne_calibrated:
            uncalibrated.append(group["name"])

    if not groups:
        print("  (IV: no run could be read; nothing to render)")
        return None

    # Every run is [cm^-3], so none is dropped -- but an uncalibrated run's
    # absolute scale rests on Aprobe and the vth convention, which a reader
    # comparing runs on one axis must be told.
    if uncalibrated:
        print(f"  (IV: ne not interferometer-calibrated for "
              f"{', '.join(uncalibrated)}; absolute scale is probe-geometry based)")

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": "Jun-2026 IV line scans - Vp, Te, ne by gas puff",
        "geometry": "line",
        # Bundle-level axis: a fallback only, since every group overrides it.
        "axis": groups[0]["axis"],
        "x": {"label": "X Position", "unit": "cm", "values": xpos},
        "groups": groups,
        "group_label": "Gas puff",
        "provenance": {
            "source": f"{len(groups)} runs, {os.path.basename(os.path.dirname(ifns[0]))}",
            "params": {"ne": _ne_provenance(uncalibrated),
                       "tips": "overlaid per frame",
                       "positions": xpos.size},
            # Every run on this page uses the same probe, so one run's tip
            # channels describe them all; the first rendered run is the sample.
            "details": _tip_details(ifns[0]),
        },
        "warning": None,
    }
    return write_slider_html(bundle, out or slider_path(name or "IV-line-all-runs"))


# Upper edge of the Isat FFT band shipped to the page.  The stored spectra run
# to Nyquist (25 MHz, 87500 bins); at 3 channels x 61 positions that is 16M
# numbers, far more than a page needs -- and the fluctuation structure sits at
# the bottom of the band.
ISAT_FMAX_KHZ = 500.0

#: How far two runs' recorded probe positions may differ and still count as the
#: same commanded line [cm]. Runs 00-06 repeat one 61-point x-line and their
#: stored coordinates differ by up to 6.3e-5 cm -- drive repeatability, not a
#: different scan. Sits far above that and far below the 1 cm point spacing, so
#: a genuinely different line still fails.
ISAT_POS_TOL_CM = 1e-3


def emit_isat_fft_slider(ifn, npz_path=None, chans=None,
                         fmax_khz=ISAT_FMAX_KHZ, fmin_khz=1.0, out=None):
    """Position-slider page of per-position Isat amplitude spectra.

    Reads the per-position spectra :func:`Jun2026_Isat.batch_fft_by_position`
    saved and renders **one panel with one trace per channel**, the slider
    scrubbing probe position.  Every Mach tip is drawn against one shared
    log-log axis, which is the comparison a multi-tip probe exists to support.

    This is the schema's non-spatial case: the *scan* axis is position and the
    panel's ``x`` is frequency.  ``axis.labels`` carries each position's
    ``(x, y)``, so the readout says where the frame is rather than "position 37".

    Args:
        ifn (str): The run's raw HDF5 path (names the page, finds the npz).
        npz_path (str): Override the co-located npz location.
        chans (list): Channel keys to draw, as stored by
            :func:`Jun2026_Isat.stored_channels`; default every stored channel.
        fmax_khz, fmin_khz (float): Band shipped to the page.  ``fmin_khz`` is
            above zero because a log frequency axis has no pixel for DC.
        out (str): Explicit output ``.html`` path.

    Returns the written ``.html`` path, or ``None`` if the run has no stored
    channel (the reason is printed, so a caller looping over runs can continue).
    """
    read = _isat_run_read(ifn, npz_path, chans, fmax_khz, fmin_khz)
    if read is None:
        return None

    run_num = run_num_of(ifn)
    run_label = jpl.run_title(ifn, run_num) or f"{run_num} Isat FFT"
    bundle = {
        "schema": SCHEMA_VERSION,
        "title": f"{run_label} \u2014 Isat FFT by position",
        "geometry": "line",
        "axis": _isat_position_axis(read.pos_x, read.pos_y),
        "x": {"label": "frequency", "unit": "kHz", "values": read.freq_khz,
              "scale": "log"},
        "fields": [_isat_field(read.traces)],
        "context": read.context,
        "provenance": {
            "source": os.path.basename(ifn),
            "params": {"band": _band_text(fmin_khz, fmax_khz, read.freq_khz),
                       "FFT window": read.window_text,
                       "positions": read.pos_x.size},
            "details": _channel_details(ifn, read.chans),
        },
        "warning": None,
    }

    name = f"{run_num}-isat-fft-slider-0to{fmax_khz:g}kHz"
    return write_slider_html(bundle, out or slider_path(name))


#: One run's prepared Isat spectra -- what both Isat emitters need from an npz.
#: The FFT window itself is not here: it survives only as the banner text and
#: the context shading, both settled inside the read.
_IsatRead = namedtuple(
    "_IsatRead", "chans traces freq_khz pos_x pos_y window_text context")


def _isat_run_read(ifn, npz_path, chans, fmax_khz, fmin_khz):
    """One run's npz -> banded traces, axes and context, or ``None``.

    Shared by the one-run and all-runs Isat pages so a change to how a run is
    prepared cannot apply to only one of them -- the rule
    :func:`_xcorr_pair_fields` follows for pairs. ``None`` (with the reason
    printed) means the run has nothing stored, so a caller looping over runs
    continues instead of dying on a run that was never analysed.
    """
    if npz_path is None:
        npz_path = jis.isat_npz_path(ifn)
    if not os.path.isfile(npz_path):
        print(f"  (Isat: no per-position npz at {npz_path})")
        return None
    if chans is None:
        chans = jis.stored_channels(npz_path)
    if not chans:
        print(f"  (Isat: no stored channel in {npz_path})")
        return None

    with np.load(npz_path) as d:
        # One page carries one frequency axis, so channels recorded on scopes of
        # different sampling rates cannot share it. The axis is stored per
        # channel; requiring them equal is what keeps a page's x-axis true of
        # every trace on it.
        freq_khz = d[f"{chans[0]}__freq"] / 1e3
        for k in chans[1:]:
            if not np.array_equal(d[f"{k}__freq"], d[f"{chans[0]}__freq"]):
                raise ValueError(
                    f"channel '{k}' has a different frequency axis than "
                    f"'{chans[0]}' (different scope sampling rates?); they "
                    "cannot share one page. Pass `chans` for one scope at a time.")
        band = (freq_khz >= fmin_khz) & (freq_khz <= fmax_khz)
        if not band.any():
            raise ValueError(f"band {fmin_khz}-{fmax_khz} kHz selects no bin of "
                             f"{freq_khz[0]:.3f}-{freq_khz[-1]:.1f} kHz")
        # Stored as (npos, nfreq), which is already the schema's (n_axis, nx).
        traces = [{"name": k, "frames": d[f"{k}__amp"][:, band]} for k in chans]
        # float32 on disk; float64 here so the cross-run position comparison is
        # not fighting the storage dtype.
        pos_x = np.asarray(d["pos_x"], float)
        pos_y = np.asarray(d["pos_y"], float)
        freq_khz = freq_khz[band]
        # Read inside the `with`: an NpzFile is a lazy zip handle, and the
        # bundle below is built after it has closed.
        window = _stored_window(d, chans)
        window_text = _window_text(set(d.files), chans, window)
        context = _fft_window_context(d, chans, window)

    return _IsatRead(chans, traces, freq_khz, pos_x, pos_y, window_text, context)


def _isat_position_axis(pos_x, pos_y):
    """The scan axis: position index, with each frame's (x, y) as its readout."""
    return {"name": "position", "unit": "",
            "values": np.arange(pos_x.size, dtype=float),
            "labels": _position_labels(pos_x, pos_y)}


def _isat_field(traces):
    """The one Isat panel: every tip overlaid on a shared log-log axis."""
    return {"name": "Isat amplitude", "unit": "V", "cmap": "viridis",
            "vmin": None, "vmax": None, "yscale": "log", "traces": traces}


def _band_text(fmin_khz, fmax_khz, freq_khz):
    return f"{fmin_khz:g}-{fmax_khz:g} kHz ({freq_khz.size} bins)"


def emit_isat_fft_slider_all(ifns, chans=None, fmax_khz=ISAT_FMAX_KHZ,
                             fmin_khz=1.0, out=None, name=None):
    """Every Isat run on **one** page: runs in the dropdown, tips overlaid.

    The counterpart to :func:`emit_iv_line_slider_all`, for runs that repeat one
    probe line while a setting is walked (runs 00-06, gas puff 23->45 ms).
    Replaces the one-file-per-run family: "how does this spectrum change with
    the puff?" becomes a dropdown rather than seven browser windows.

    The dropdown says the *setting*, not the run number -- the puff duration is
    the knob these runs vary, so ``"35 ms (run 03)"``; entries are ordered by it
    rather than by run, which is why run 06 (23 ms) leads. Runs with no puff in
    their description fall back to the run number alone and sort last.

    Args:
        ifns: The runs' raw HDF5 paths. Each needs its ``-isat-fft-data.npz``
            beside it (:func:`Jun2026_Isat.batch_fft_by_position`).
        chans (list): Channel keys to draw; default every channel stored by the
            run being read, so runs need not agree on their tips.
        fmax_khz, fmin_khz (float): Band shipped to the page.
        out, name (str): Explicit output path / output stem.

    Returns the written ``.html`` path, or ``None`` when no run could be read.

    Raises ``ValueError`` if the runs disagree on their frequency axis or probe
    line beyond :data:`ISAT_POS_TOL_CM` -- one page carries one of each, and
    drawing a second run's spectra against the first's axes would mislabel every
    frame with no visible symptom.
    """
    groups, shared, ref = [], None, None
    for ifn in ifns:
        read = _isat_run_read(ifn, None, chans, fmax_khz, fmin_khz)
        if read is None:
            continue

        label, sort_key = _puff_run_entry(ifn, run_num_of(ifn))

        # One page carries one frequency axis and one probe line for every run.
        # Checked rather than assumed: a page that adopted the first run's axes
        # and drew another run's data against them would be wrong invisibly.
        if shared is None:
            shared, ref = read, label
        else:
            if not np.array_equal(read.freq_khz, shared.freq_khz):
                raise ValueError(
                    f"{label} has a different frequency axis than {ref} "
                    "(different scope sampling rate or FFT length); they "
                    "cannot share one page's x-axis.")
            # Tolerant, not exact: see ISAT_POS_TOL_CM. A run measured at a
            # different *number* of positions is a different scan outright.
            same_line = (read.pos_x.shape == shared.pos_x.shape
                         and np.allclose(read.pos_x, shared.pos_x,
                                         rtol=0, atol=ISAT_POS_TOL_CM)
                         and np.allclose(read.pos_y, shared.pos_y,
                                         rtol=0, atol=ISAT_POS_TOL_CM))
            if not same_line:
                raise ValueError(
                    f"{label} was measured on a different probe line than "
                    f"{ref}; they cannot share one page's position axis.")

        # Paired with the sort key rather than carrying it as a group field: a
        # group is validated against the schema, and a stray key would have to
        # be popped back off before rendering.
        # Its own raw record, so the figure above the panels tracks the dropdown
        # rather than showing one run's trace over every run's spectra.
        groups.append((sort_key,
                       {"name": label, "fields": [_isat_field(read.traces)],
                        "context": _context_for_run(read.context, label)}))

    if not groups:
        print("  (Isat: no run could be read; nothing to render)")
        return None

    # By the walked setting, not by run number: the dropdown reads as the sweep
    # it is (23, 24, 25, 30 ms...), which run order would scramble.
    groups = [group for _, group in sorted(groups, key=lambda pair: pair[0])]

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": "Jun-2026 Isat FFT by position - runs by gas puff",
        "geometry": "line",
        # Every run shares these, checked above; the first read defines them.
        "axis": _isat_position_axis(shared.pos_x, shared.pos_y),
        "x": {"label": "frequency", "unit": "kHz", "values": shared.freq_khz,
              "scale": "log"},
        "groups": groups,
        "group_label": "Gas puff",
        # No bundle-level context: each group carries its own record, so a
        # shared fallback could only ever be one run's trace standing in for
        # another's.
        "provenance": {
            "source": f"{len(groups)} runs, "
                      f"{os.path.basename(os.path.dirname(ifns[0]))}",
            "params": {"band": _band_text(fmin_khz, fmax_khz, shared.freq_khz),
                       "FFT window": shared.window_text,
                       "positions": shared.pos_x.size,
                       "runs": len(groups)},
            # Every run on this page carries the same tips, so the first
            # rendered run's channels describe them all.
            "details": _channel_details(ifns[0], shared.chans),
        },
        "warning": None,
    }

    name = name or f"isat-fft-all-runs-0to{fmax_khz:g}kHz"
    return write_slider_html(bundle, out or slider_path(name))


def _context_for_run(context, run_label):
    """One run's context block, titled with the run it came from.

    The figure follows the dropdown, so the title is not what identifies the
    record -- but the caption is the only place the run's name appears while a
    given group is selected, and a reader who has scrolled past the dropdown
    would otherwise have to scroll back to know which run they are looking at.

    A copy, not a mutation: ``read.context`` belongs to the caller's
    :class:`_IsatRead`, which the single-run emitter also hands out.
    """
    if context is None:
        return None
    # .get: the schema does not require a context title, and a page whose
    # context lost its run name is a worse failure than a bare run name.
    title = context.get("title", "")
    return {**context, "title": f"{title} \u2014 {run_label}" if title else run_label}


def _window_text(d_files, chans, window):
    """Banner text for the FFT window: the value, 'mixed', or 'not recorded'."""
    if window:
        return f"{window[0]:g}-{window[1]:g} ms"
    if all(f"{k}__tmin_ms" in d_files for k in chans):
        return "mixed (channels analysed over different windows)"
    return "not recorded"


def _stored_window(d, chans):
    """The FFT window ``[tmin_ms, tmax_ms]`` these channels share, else ``None``.

    Stored per channel because it is a per-call argument: a second
    :func:`Jun2026_Isat.batch_fft_by_position` call with a different window
    leaves earlier channels' spectra untouched, so one page can hold two. The
    banner and the shaded span state one window, so they may only do so when
    the channels agree -- otherwise the page says "mixed" and shades nothing,
    rather than labelling half its traces with a window they never saw.
    """
    windows = set()
    for k in chans:
        if f"{k}__tmin_ms" not in d.files:
            return None                      # npz predates per-channel windows
        windows.add((float(d[f"{k}__tmin_ms"]), float(d[f"{k}__tmax_ms"])))
    if len(windows) != 1:
        return None
    return list(windows.pop())


def _channel_details(ifn, chans):
    """``{'machscope/C2': 'MP@P33, Isat-X-'}`` for the banner's second line.

    The stored key names where a trace came from in the npz; the run's own
    description names what the tip actually measured, which is what identifies
    it on the page. Falls back to the bare key for any channel the file does not
    describe, so a page never loses a trace to a missing description.

    ``chans`` entries are ``(scope, chan)`` tuples or ``"scope-chan"`` strings
    -- the xcorr emitters carry pairs as tuples and the Isat one keys its npz by
    string, and neither spelling is worth converting at the call site. Repeated
    channels collapse (a pair list names each end several times) and order is
    kept, so the line reads in the order the page introduces them.
    """
    try:
        desc = open_lapd(ifn).channel_descriptions()
    except (NotImplementedError, OSError):
        return {}                       # bapsflib/legacy file, or unreadable
    out = {}
    for entry in chans:
        scope, chan = (entry if isinstance(entry, tuple)
                       else entry.rpartition("-")[::2])
        # "scope/chan": the spelling pair_label already uses for a channel.
        out.setdefault(f"{scope}/{chan}", desc.get(scope, {}).get(chan)
                       or f"{scope}/{chan}")
    return out


def _tip_details(ifn):
    """``{'tip P29-L': 'I lpscope/C1 \u2014 "I, LP@P29-L"; V lpscope/C2 \u2014 ...'}``.

    The IV page's counterpart to :func:`_channel_details`. A tip is a *fitted*
    quantity, not a channel: Vp/Te/ne come from an I+V sweep pair, so what
    identifies it is both channels and what the run says each of them was.
    Returns ``{}`` when the run has no sweep npz, which is what the caller ships
    for a run that was never processed.

    Keyed off the **sweep npz**, not a fresh discovery pass over the raw file:
    the npz records the channels analysis actually ran on, so a run processed
    under a ``port=`` override (:func:`Jun2026_IV.process_run`) is named here by
    the port it was analyzed as. Re-deriving from the descriptions would label
    the page with the port they claim while every trace on it carries the other
    -- a caption that looks authoritative and disagrees with the data.

    The description text still comes from the raw file, since that is the only
    record of what each scope channel was; it is shown next to the channel it
    describes, so a description naming a different port stays visible rather
    than being silently corrected.
    """
    from data_analysis.plasma.langmuir import sweep_npz_paths

    sweep_path, _ = sweep_npz_paths(os.path.dirname(ifn), run_num_of(ifn))
    if not os.path.isfile(sweep_path):
        return {}
    try:
        desc = open_lapd(ifn).channel_descriptions()
    except (NotImplementedError, OSError):
        desc = {}

    scope = jiv.SCOPE_NAME or jiv.find_lp_scope(ifn)
    out = {}
    with np.load(sweep_path) as d:
        for tip in (str(c) for c in d["channels"]):
            parts = []
            for role in ("I", "V"):
                key = f"{tip}/{role}_chan"
                if key not in d.files:
                    continue
                chan = str(d[key])
                text = desc.get(scope, {}).get(chan)
                parts.append(f"{role} {scope}/{chan}"
                             + (f" \u2014 {text}" if text else ""))
            if parts:
                out[f"tip {tip}"] = "; ".join(parts)
    return out


def _fft_window_context(d, chans, window):
    """The page's static figure: the raw record with the FFT window shaded.

    Built from the decimated trace :func:`Jun2026_Isat.batch_fft_by_position`
    stores per channel; ``window`` is the ``(tmin_ms, tmax_ms)`` pair to shade,
    or ``None``. Returns ``None`` for an npz predating those arrays -- the page
    omits the figure rather than failing, so an older npz still re-renders.

    The traces are the raw record at ONE position, while the panels below are
    per-position spectra; the figure answers "what was analysed", not "what does
    this position look like", which is why it does not follow the slider.
    """
    # Channel keys are "<scope>-<chan>", and the time axis is stored once for
    # the scope they share.
    scope_key = f"{chans[0].rsplit('-', 1)[0]}__raw_t"
    if f"{chans[0]}__raw" not in d.files or scope_key not in d.files:
        return None
    return {
        "title": "raw signal, first position (shot 0)",
        # Stored in seconds (the run's own time axis); the page reads ms.
        "x": {"label": "time", "unit": "ms", "values": d[scope_key] * 1e3},
        "y": {"label": "signal", "unit": "V"},
        "traces": [{"name": k, "values": d[f"{k}__raw"]} for k in chans],
        "span": window,
        # The numbers are in the banner; repeating them on the shading would
        # state one fact three times on one page.
        "span_label": "FFT window",
    }


def _position_labels(pos_x, pos_y):
    """Per-position readout text: ``'(x, y) = (-30.0, 0.0) cm'``.

    Always both coordinates, including on a line scan where one of them never
    moves. Runs 00-06 scan x only, but the probe drives are 2-D and a later run
    on the same pipeline may scan y or both; a label that named only the moving
    axis would then quietly describe the wrong scan, and the reader has no way
    to tell which convention a given page used.
    """
    # The literal "0.0", rather than formatting: the held axis of a line scan is
    # not exactly constant (float32 y varies by ~2e-15 cm), and f"{-1e-16:.1f}"
    # is "-0.0", which reads as a real coordinate.
    fmt = lambda v: f"{v:.1f}" if abs(v) >= 0.05 else "0.0"
    return [f"(x, y) = ({fmt(x)}, {fmt(y)}) cm" for x, y in zip(pos_x, pos_y)]


def emit_isat_fft_sliders(ifns, **kwargs):
    """:func:`emit_isat_fft_slider` for several runs; returns the written paths.

    One page *per run*, skipping runs with nothing stored. Reach for
    :func:`emit_isat_fft_slider_all` instead where the runs share a probe line:
    it puts them behind one dropdown, which is what makes them comparable.
    This one is for runs that do not -- different grids, or different scopes --
    and so cannot share a page's axes.
    """
    return [p for p in (emit_isat_fft_slider(f, **kwargs) for f in ifns) if p]


def _ne_provenance(uncalibrated):
    """Banner text for the ne on this page: [cm^-3] either way, but scaled how.

    Names the runs whose absolute scale is probe-geometry based rather than
    interferometer-backed -- the one thing the [cm^-3] label cannot say.
    """
    if not uncalibrated:
        return "interferometer-calibrated [cm^-3]"
    return (f"[cm^-3]; probe-geometry scale (no interferometer) for "
            f"{', '.join(uncalibrated)}")


def _puff_run_entry(ifn, run_num):
    """A run's dropdown entry -> ``(label, sort_key)``, from one read of the file.

    ``('35 ms  (run 03)', (0, 35.0))``, or ``('run 03', (1, 3.0))`` where the
    description records no puff. Both *_all pages label their dropdown this way,
    so the format is declared here rather than once per page.

    The puff duration is the knob these runs vary, so it leads the label and
    orders the entries; runs without one sort after those with one, by run
    number, since interleaving them at an invented duration would put a wrong
    reading on the dropdown.

    Read via :func:`data_analysis.io.parse_gas_puff`, the one declaration of the
    operator's puff phrasing, rather than by regexing a formatted title back
    apart. That also avoids :func:`Jun2026_plot.puff_title`'s no-puff fallback,
    which returns the bare run number -- a label of "24" that reads as a
    duration but is a run number.
    """
    desc = jpl._run_description(ifn)
    puff = parse_gas_puff(desc.raw) if desc is not None else None
    if not puff:
        return f"run {run_num}", (1, float(run_num))
    puff_ms = puff[1]
    return f"{puff_ms:g} ms  (run {run_num})", (0, puff_ms)


# Every line-scan run in the data directory, in run order.  Deliberately broader
# than the runs that have saved sweep npz: emit_iv_line_slider_all skips (and
# names) the ones that do not, so a run processed later joins the page without
# anyone remembering to widen a pattern here.
IV_LINE_GLOB = "*-line_*.hdf5"


if __name__ == "__main__":
    emit_xcorr_slider_all(jxc.IFN)
    emit_iv_line_slider_all(
        sorted(glob.glob(os.path.join(jis.DATA_DIR, IV_LINE_GLOB))))
