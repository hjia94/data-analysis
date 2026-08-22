"""Jun-2026 slider *adapters*: saved analysis npz -> a scrubbable HTML page.

The third piece of the Jun-2026 split. ``Jun2026_xcorr`` (and friends) read raw
HDF5 and save arrays; ``Jun2026_plot`` draws static publication figures; this
module builds :mod:`data_analysis.viz.slider_html` bundles from those same saved
arrays and renders a self-contained interactive page.

It is **additive and read-only**: nothing here changes the analysis modules, the
static figures, or the co-located data ``.npz`` files. A slider page is a
preliminary-data tool -- scrub the scan axis to find the interesting frame, then
draw that frame properly with ``Jun2026_plot``.

Why an adapter per diagnostic
--------------------------------------------------------------------------
Only this layer knows an experiment's storage layout -- xcorr's multi-pair key
prefixes here, the IV pipeline's per-tip file pair for a future
``emit_iv_slider``. The renderer downstream knows none of it: it takes a
validated bundle of prepared frames (see the schema in
:mod:`data_analysis.viz.slider_html`) and nothing else. Adding a diagnostic is
one more small function here, against a frozen schema.

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
:func:`emit_xcorr_slider_all` puts **every pair of a run on one page** behind a
dropdown, which is the one to reach for when the question is "which channel pair
shows this?" -- comparing pairs by switching a dropdown at a held frequency,
rather than by tiling browser windows. The two share
:func:`_xcorr_pair_fields`, so a change to how a pair is prepared cannot apply
to only one of them.
"""

import os

import numpy as np

from data_analysis.io import open_lapd, parse_gas_puff
from data_analysis.plasma.langmuir import load_plasma_data, load_sweep_axes
from data_analysis.utils import run_num_of
from data_analysis.viz.plot_utils import fig_path, grid_frames
from data_analysis.viz.slider_html import SCHEMA_VERSION, write_slider_html

import Jun2026_xcorr as jxc
import Jun2026_plot as jpl
import Jun2026_Isat as jis
import Jun2026_IV as jiv


# Default subdirectory under $DATA_ANALYSIS_OUTPUT/figures/ -- the same place
# the static Jun-2026 PNGs go; a slider page is a render, not data.
FIG_SUBDIR = jpl.FIG_SUBDIR

# Upper edge of the frequency band shipped to the page, kHz.  Run 26's measured
# coherence all lies below ~10 kHz (strongest bins 4.88 / 3.66 / 8.54 / 2.44 kHz),
# so 20 kHz covers every real feature; 50 kHz would add ~24 bins of near-zero
# coherence -- payload and scrub distance spent on noise.  An emit-time
# parameter, recorded in the page's provenance banner.
XCORR_FMAX_KHZ = 20.0


#: Shown when no coherence floor was applied.  Held here rather than inlined
#: because both the one-pair and the all-pairs page must carry the identical
#: caveat -- an unmasked phase map that does not say it is unmasked is a trap.
_XCORR_WARNING = (
    "No coherence floor applied — cross-phase is random noise wherever γ² is "
    "near zero. Read the phase map only where the coherence map beside it is high.")


def slider_path(name, subdir=FIG_SUBDIR):
    """Centralized ``.html`` location: :func:`data_analysis.viz.plot_utils.fig_path`
    with this campaign's default subdirectory and an ``.html`` extension.

    Pages land under ``$DATA_ANALYSIS_OUTPUT/figures/<subdir>/`` beside the PNGs
    -- a slider page is a render, not data -- never next to the raw data.
    """
    return fig_path(name, subdir, ext=".html")


def _xcorr_params(freq_khz, fmax_khz, gamma2_floor):
    """The provenance ``params`` block shared by both xcorr pages."""
    params = {"band": f"0-{fmax_khz:g} kHz ({freq_khz.size} bins)",
              "window": f"{jxc.TMIN_MS}-{jxc.TMAX_MS} ms",
              "nperseg": jxc.NPERSEG}
    if gamma2_floor is not None:
        params["phase masked below gamma2"] = gamma2_floor
    return params


def _xcorr_pair_fields(ifn, ch_a, ch_b, npz_path, fmax_khz, gamma2_floor):
    """Prepare one channel pair into slider *fields*: band-limit, gate, grid.

    The single place a stored pair becomes drawable frames, so the one-pair page
    and one channel of the all-pairs page cannot end up preparing the same data
    differently.  Returns ``(fields, freq_khz, xs, ys)``, or ``None`` (having
    printed why) when the pair has no saved entry.
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
    params = _xcorr_params(freq_khz, fmax_khz, gamma2_floor)

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

        # One page carries one axis and one probe plane for every channel, so
        # the pairs must agree on them.  They come from the same npz and always
        # do -- but a page that silently adopted one pair's axis and labelled
        # another pair's data with it would be wrong in a way no reader could
        # see, so it is checked rather than assumed.
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
                       "params": _xcorr_params(freq_khz, fmax_khz, gamma2_floor),
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


def _iv_run_group(data_dir, run_num, tips, calibrated=True):
    """One run as a slider *group*: per-tip traces of each field, vs position.

    Returns ``(group, xpos)``, or ``None`` when the run cannot be read. Each
    field carries one trace per tip, so a frame shows both tips of the same
    quantity against one y-axis -- the comparison the static figure could only
    make by drawing separate lines per timestamp.
    """
    traces_by_field = {name: [] for name, _ in IV_FIELDS}
    axis_values, xpos = None, None

    for tip in tips:
        try:
            Vp, Te, ne, *_errs, t_ls = load_plasma_data(data_dir, run_num, tip=tip)
            xs, *_ = load_sweep_axes(data_dir, run_num, tip=tip)
        except (FileNotFoundError, OSError, KeyError) as exc:
            print(f"  (IV: run {run_num} tip {tip} unreadable: {exc})")
            return None
        ne = jpl._load_ne(data_dir, run_num, tip, ne, calibrated)

        # Some runs store one more timestamp than they have sweeps. The static
        # figure indexes t_ls[t_idx] for t_idx < n_sweeps, i.e. it uses the
        # leading timestamps and ignores the trailing one; match that exactly
        # so the page and the PNG label the same frame with the same time.
        n_sweeps = Vp.shape[1]
        times_ms = np.asarray(t_ls, float)[:n_sweeps] * 1e3

        # Tips of one run are the same acquisition, so they agree on the time
        # axis and the probe line. Checked rather than assumed: silently
        # adopting one tip's axis for the other's data would mislabel every
        # frame with no visible symptom. Checked *before* accumulating, so a
        # rejected tip never leaves half-built traces behind.
        #
        # Compared to a tolerance, not bit-for-bit: the two tips are timed
        # independently and their recorded sweep times differ by roughly
        # IV_TIP_JITTER_MS. That is the same instant recorded twice, so
        # requiring equality would reject every real run; a *sweep* apart is a
        # different acquisition and still fails.
        if axis_values is None:
            axis_values, xpos, first_tip = times_ms, np.asarray(xs, float), tip
        else:
            same_time = (times_ms.shape == axis_values.shape
                         and np.allclose(times_ms, axis_values, rtol=0,
                                         atol=IV_TIP_JITTER_MS))
            if not (same_time and np.array_equal(np.asarray(xs, float), xpos)):
                print(f"  (IV: run {run_num} tip {tip} disagrees with tip "
                      f"{first_tip} on the time axis or probe line; skipping "
                      "the run)")
                return None

        # Saved as (npos, n_sweeps); the schema wants (n_axis, nx). Zipped
        # against IV_FIELDS so the panel order is stated once.
        for (name, _), cube in zip(IV_FIELDS,
                                   (Vp.T, Te.T, ne.T, (Te * ne).T)):
            traces_by_field[name].append({"name": f"tip {tip}", "frames": cube})

    group = {
        "name": "",                      # filled in by the caller (delay time)
        # The first tip's timestamps label the frames; the check above has
        # established the others match to well within a sweep.
        "axis": {"name": "time", "unit": "ms", "values": axis_values},
        "fields": [{"name": name, "unit": unit, "cmap": "viridis",
                    # No natural physical scale, and the per-frame range varies
                    # by ~10x across a run, so these follow the page's toggle.
                    "vmin": None, "vmax": None,
                    "traces": traces_by_field[name]}
                   for name, unit in IV_FIELDS],
    }
    return group, xpos


def emit_iv_line_slider_all(ifns, out=None, calibrated=True, name=None):
    """Every IV run on **one** page: runs in the dropdown, tips overlaid.

    Replaces the ``IV_line_*.png`` family, where each figure fixed a handful of
    timestamps and each run needed its own file. Here the slider scrubs the
    timestamps and the dropdown switches runs, so "how does this profile evolve,
    and how does that differ between fill delays?" is two controls rather than
    seven files.

    Args:
        ifns: The run ``.hdf5`` paths, in dropdown order. Each needs its saved
            ``-tip*-plasma-data.npz`` beside it (:func:`Jun2026_IV.process_run`).
        out: Destination ``.html``; defaults under :func:`slider_path`.
        calibrated (bool): Prefer interferometer-calibrated ne, falling back to
            raw with a note -- the rule :func:`Jun2026_plot._load_ne` applies.
        name (str): Output stem, when the default is not wanted.

    Runs keep their own time axes: they differ in both sweep count and span, so
    the slider holds the frame *index* across a switch and the readout shows the
    real time for the run now selected.
    """
    groups, xpos, shared_label = [], None, None
    for ifn in ifns:
        data_dir = os.path.dirname(ifn)
        run_num = run_num_of(ifn)
        tips = [t for t in jpl.discover_tips(ifn) if t != "override"]
        if not tips:
            print(f"  (IV: run {run_num} has no saved tip data; skipping)")
            continue

        prepared = _iv_run_group(data_dir, run_num, tips, calibrated)
        if prepared is None:
            continue
        group, xs = prepared

        # The delay time is the knob that distinguishes these runs, so it is
        # what the dropdown says -- "23 ms", not "run 24".
        delay = _delay_label(ifn)
        group["name"] = f"{delay}  (run {run_num})" if delay else f"run {run_num}"

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

    if not groups:
        print("  (IV: no run could be read; nothing to render)")
        return None

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": "Jun-2026 IV line scans - Vp, Te, ne by fill delay",
        "geometry": "line",
        # Bundle-level axis: a fallback only, since every group overrides it.
        "axis": groups[0]["axis"],
        "x": {"label": "X Position", "unit": "cm", "values": xpos},
        "groups": groups,
        "provenance": {
            "source": f"{len(groups)} runs, {os.path.basename(os.path.dirname(ifns[0]))}",
            "params": {"ne": "interferometer-calibrated" if calibrated else "raw",
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
        pos_x, pos_y = d["pos_x"], d["pos_y"]
        freq_khz = freq_khz[band]
        # Read inside the `with`: an NpzFile is a lazy zip handle, and the
        # bundle below is built after it has closed.
        window = ([float(d["tmin_ms"]), float(d["tmax_ms"])]
                  if "tmin_ms" in d.files else None)
        context = _fft_window_context(d, chans, window)

    run_num = run_num_of(ifn)
    run_label = jpl.run_title(ifn, run_num) or f"{run_num} Isat FFT"
    bundle = {
        "schema": SCHEMA_VERSION,
        "title": f"{run_label} \u2014 Isat FFT by position",
        "geometry": "line",
        "axis": {"name": "position", "unit": "",
                 "values": np.arange(pos_x.size, dtype=float),
                 "labels": _position_labels(pos_x, pos_y)},
        "x": {"label": "frequency", "unit": "kHz", "values": freq_khz,
              "scale": "log"},
        "fields": [{"name": "Isat amplitude", "unit": "V", "cmap": "viridis",
                    "vmin": None, "vmax": None, "yscale": "log",
                    "traces": traces}],
        "context": context,
        "provenance": {
            "source": os.path.basename(ifn),
            "params": {"band": f"{fmin_khz:g}-{fmax_khz:g} kHz "
                               f"({freq_khz.size} bins)",
                       "FFT window": (f"{window[0]:g}-{window[1]:g} ms"
                                      if window else "not recorded"),
                       "positions": pos_x.size},
            "details": _channel_details(ifn, chans),
        },
        "warning": None,
    }

    name = f"{run_num}-isat-fft-slider-0to{fmax_khz:g}kHz"
    return write_slider_html(bundle, out or slider_path(name))


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
    """``{'tip L': 'I lpscope/C1 "Isat, LP@P29-L" + V lpscope/C2 ...'}``.

    The IV page's counterpart to :func:`_channel_details`. A tip is a *fitted*
    quantity, not a channel: Vp/Te/ne come from an I+V sweep pair, so what
    identifies it is both channels and what the run says each of them was.
    Returns ``{}`` when the map or the descriptions cannot be read, which is
    what the caller ships when a run predates description-tagged channels.
    """
    try:
        chan_map = jiv.resolve_iv_channel_map(ifn)
        desc = open_lapd(ifn).channel_descriptions()
    except (NotImplementedError, OSError, ValueError, FileNotFoundError):
        return {}
    out = {}
    for tip, (scope, i_chan, v_chan) in chan_map.items():
        parts = []
        for role, chan in (("I", i_chan), ("V", v_chan)):
            text = desc.get(scope, {}).get(chan)
            parts.append(f"{role} {scope}/{chan}"
                         + (f" \u2014 {text}" if text else ""))
        out[f"tip {tip}" if tip else "tip"] = "; ".join(parts)
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

    One page per run (each run has its own probe line and npz), skipping runs
    with nothing stored.
    """
    return [p for p in (emit_isat_fft_slider(f, **kwargs) for f in ifns) if p]


def _delay_label(ifn):
    """A run's fill delay as a dropdown entry: ``'23 ms'``.

    Read from the parsed description rather than by regexing a formatted title
    back apart: :func:`data_analysis.io.parse_gas_puff` is the one declaration
    of the operator's puff phrasing, and going through it also avoids
    :func:`Jun2026_plot.puff_title`'s no-puff fallback, which returns the bare
    run number -- a label of "24" that reads as a delay but is a run number.
    Returns ``""`` when there is no puff line, so the caller shows the run
    alone instead of a wrong delay.
    """
    desc = jpl._run_description(ifn)
    puff = parse_gas_puff(desc.raw) if desc is not None else None
    return f"{puff[1]:g} ms" if puff else ""


if __name__ == "__main__":
    emit_xcorr_slider_all(jxc.IFN)
