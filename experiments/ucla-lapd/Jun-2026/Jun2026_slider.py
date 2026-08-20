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
:func:`emit_xcorr_slider` renders one channel pair per page.
:func:`emit_xcorr_slider_all` puts **every pair of a run on one page** behind a
dropdown, which is the one to reach for when the question is "which channel pair
shows this?" -- comparing pairs by switching a dropdown at a held frequency,
rather than by tiling browser windows. The two share
:func:`_xcorr_pair_fields`, so a change to how a pair is prepared cannot apply
to only one of them.
"""

import os

import numpy as np

from data_analysis.utils import run_num_of
from data_analysis.viz.plot_utils import fig_path, grid_frames
from data_analysis.viz.slider_html import SCHEMA_VERSION, write_slider_html

import Jun2026_xcorr as jxc
import Jun2026_plot as jpl


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
        "provenance": {"source": os.path.basename(ifn), "params": params},
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

    groups, shared = [], None
    for ch_a, ch_b in pairs:
        prepared = _xcorr_pair_fields(ifn, ch_a, ch_b, npz_path, fmax_khz,
                                      gamma2_floor)
        if prepared is None:      # missing entry: already reported, skip it
            continue
        fields, *axes = prepared
        label = jxc.pair_label(ch_a, ch_b)

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
                       "params": _xcorr_params(freq_khz, fmax_khz, gamma2_floor)},
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


if __name__ == "__main__":
    emit_xcorr_slider_all(jxc.IFN)
