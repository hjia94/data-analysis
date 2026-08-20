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


def slider_path(name, subdir=FIG_SUBDIR):
    """Centralized ``.html`` location: :func:`data_analysis.viz.plot_utils.fig_path`
    with this campaign's default subdirectory and an ``.html`` extension.

    Pages land under ``$DATA_ANALYSIS_OUTPUT/figures/<subdir>/`` beside the PNGs
    -- a slider page is a render, not data -- never next to the raw data.
    """
    return fig_path(name, subdir, ext=".html")


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

    # Optional coherence gate on the phase map only (see the docstring).
    if gamma2_floor is not None:
        phase_deg = np.where(gamma2 < gamma2_floor, np.nan, phase_deg)

    g2_frames, xs, ys = grid_frames(pos_x, pos_y, gamma2)
    ph_frames, _, _ = grid_frames(pos_x, pos_y, phase_deg)

    run_num = run_num_of(ifn)
    # The same descriptive title the static figure uses (run number + gas-puff
    # setting, the knob that changes run to run), so a page and its PNG are
    # recognizably the same run.
    run_label = jpl.run_title(ifn, run_num) or f"{run_num} xcorr plane"
    pair_label = f"{ch_a[0]}/{ch_a[1]} vs {ch_b[0]}/{ch_b[1]}"
    params = {"band": f"0-{fmax_khz:g} kHz ({freq_khz.size} bins)",
              "window": f"{jxc.TMIN_MS}-{jxc.TMAX_MS} ms",
              "nperseg": jxc.NPERSEG}
    if gamma2_floor is not None:
        params["phase masked below gamma2"] = gamma2_floor

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": f"{run_label} — {pair_label}",
        "geometry": "plane",
        "axis": {"name": "frequency", "unit": "kHz", "values": freq_khz},
        "x": {"label": "X position", "unit": "cm", "values": xs},
        "y": {"label": "Y position", "unit": "cm", "values": ys},
        "fields": [
            {"name": "coherence γ²", "unit": "", "frames": g2_frames,
             "cmap": "viridis", "vmin": 0.0, "vmax": 1.0},
            {"name": "cross-phase Δφ", "unit": "deg", "frames": ph_frames,
             "cmap": "twilight", "vmin": -180.0, "vmax": 180.0},
        ],
        "provenance": {"source": os.path.basename(ifn), "params": params},
        "warning": (None if gamma2_floor is not None else
                    "No coherence floor applied — cross-phase is random noise "
                    "wherever γ² is near zero. Read the phase map only where the "
                    "coherence map beside it is high."),
    }

    name = (f"{run_num}-xcorr-slider-{ch_a[0]}{ch_a[1]}-{ch_b[0]}{ch_b[1]}"
            f"-0to{fmax_khz:g}kHz")
    return write_slider_html(bundle, out or slider_path(name))


def emit_xcorr_sliders(ifn, npz_path=None, **kwargs):
    """Render a page for **every** channel pair stored in a run's xcorr npz.

    A run's npz accumulates pairs under separate key prefixes, so this asks
    :func:`Jun2026_xcorr.stored_pairs` what is actually stored rather than
    making the caller name each pair.  Returns the list of written paths.
    """
    if npz_path is None:
        npz_path = jxc.xcorr_npz_path(ifn)

    paths = []
    for key in jxc.stored_pairs(npz_path):
        ch_a, ch_b = jxc._pair_from_key(key)
        path = emit_xcorr_slider(ifn, ch_a, ch_b, npz_path=npz_path, **kwargs)
        if path is not None:
            paths.append(path)
    return paths


if __name__ == "__main__":
    emit_xcorr_sliders(jxc.IFN)
