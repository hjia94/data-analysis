"""Jun-2026 LAPD cross-correlation analysis between two scope channels.

Two channels, three frequency-domain relationships:

1. magnitude-squared coherence ``gamma2(f)`` -- shared frequency bands (0..1)
2. cross-phase ``phase(f)`` -- phase difference per frequency
3. time-lag cross-correlation whose peak is the scalar time delay.

Channel identity is a ``(scope, channel)`` pair, e.g. ``("machscope", "C3")`` --
a run has several scope groups, so the scope must be named.  Channel *inspection*
(which scope/channel is what) is not done here; pick the pair yourself.

Same read path as ``Jun2026_IV`` / ``Jun2026_Isat``: channels via
``run.channel(name, scope_name=...)``; positions via
``Jun2026_IV.read_lp_positions``.  Signal is kept raw (volts) -- the *shape* of
the coherence/phase is what matters, not absolute current scaling.
"""

import os
import numpy as np

from data_analysis.io import open_lapd
from data_analysis.signal import (
    cross_correlation,
    coherence_spectrum,
    cross_phase_spectrum,
    avg_cross_spectrum,
    band_cross_spectrum,
    peak_cross_spectrum,
)

import Jun2026_IV as jiv


# --- Configuration (edit-in-place, like Jun2026_Isat.py) --------------------
IFN = r"D:\data\LAPD\jun2026-jia\07-He-800G-bias40V-Isat-p29-plane_2026-06-10.hdf5"
CH_A        = ("machscope", "C2")      # e.g. LP@P29 Isat-L
CH_B        = ("machscope", "C3")      # e.g. LP@P29 Isat-R
TMIN_MS, TMAX_MS = 1.5, 5.0            # analysis time window (ms)
NPERSEG     = 4096                     # Welch segment length (freq res vs variance)
FBAND_KHZ   = (10.0, 14.0)            # fixed narrow band for band_xcorr maps (kHz)
# Peak-tracking (for a mode whose frequency drifts across the plane): find the
# per-position coherence peak inside SEARCH_KHZ, then integrate +-DELTAF_KHZ.
SEARCH_KHZ  = (5.0, 30.0)             # window to locate the per-position peak (kHz)
DELTAF_KHZ  = 2.0                     # half-width integrated around each peak (kHz)


# =========================================================================== #
#  Reading -- pull a position's shots for both channels onto one time grid
# =========================================================================== #

def _clip_window(tarr, tmin_ms, tmax_ms):
    """Index range ``[i0, i1)`` of ``tarr`` (seconds) inside ``[tmin, tmax]`` ms."""
    i0, i1 = np.searchsorted(tarr, [tmin_ms * 1e-3, tmax_ms * 1e-3])
    if i1 - i0 < 2:
        raise ValueError(
            f"window {tmin_ms}-{tmax_ms} ms selects < 2 samples (i0={i0}, i1={i1})")
    return int(i0), int(i1)


def _finite_row_mask(stack_a, stack_b):
    """Boolean mask of shots (rows) that are all-finite in *both* stacks."""
    return (np.all(np.isfinite(stack_a), axis=1)
            & np.all(np.isfinite(stack_b), axis=1))


def _read_pair_at_position(run, ch_a, ch_b, npos, nshot, pos_index,
                           tmin_ms=TMIN_MS, tmax_ms=TMAX_MS):
    """Read both channels' shots at one probe position onto a common time grid.

    ``ch_a`` / ``ch_b`` are ``(scope, channel)`` pairs.  Only that position's
    ``nshot`` shots are read off disk (a positional shot slice, same ordering as
    ``Jun2026_Isat.get_isat_at_position``).  Returns ``(stack_a, stack_b, dt)``
    with the two ``(nshot, nsamples)`` stacks clipped to ``[tmin_ms, tmax_ms]``
    and on the *same* grid, plus the sample interval ``dt`` (seconds).

    Same-scope pair (the common case): both channels share the identical scope
    ``tarr`` recorded in the HDF5, so it is used directly -- no resampling.
    Cross-scope pair (secondary): the two scopes can differ in dt/t0, so channel
    B's rows are resampled onto A's clipped grid via ``np.interp``.
    """
    (scope_a, chan_a), (scope_b, chan_b) = ch_a, ch_b
    shots = slice(pos_index * nshot, (pos_index + 1) * nshot)

    stack_a, tarr_a = run.channel(chan_a, scope_name=scope_a, shots=shots)
    stack_b, tarr_b = run.channel(chan_b, scope_name=scope_b, shots=shots)
    if stack_a is None or stack_b is None:
        raise ValueError(f"could not read {ch_a} or {ch_b} at position {pos_index}")

    ia0, ia1 = _clip_window(tarr_a, tmin_ms, tmax_ms)
    ta = tarr_a[ia0:ia1]
    sa = stack_a[:, ia0:ia1]
    dt = ta[1] - ta[0]

    if scope_a == scope_b:
        # Same scope -> identical time array; clip B with the same indices.
        sb = stack_b[:, ia0:ia1]
        return sa, sb, dt

    # Cross scope: clip B to its own window, then interpolate each row onto A's
    # grid (row-wise) so both stacks live on the same axis for the FFTs.
    ib0, ib1 = _clip_window(tarr_b, tmin_ms, tmax_ms)
    tb = tarr_b[ib0:ib1]
    sb_clip = stack_b[:, ib0:ib1]
    sb = np.vstack([np.interp(ta, tb, row) for row in sb_clip])
    return sa, sb, dt


# =========================================================================== #
#  Analysis -- per-shot and ensemble-averaged
# =========================================================================== #

def xcorr_per_shot(stack_a, stack_b, dt, nperseg=NPERSEG):
    """The three correlation methods computed **per shot** (for the overlay figure).

    ``stack_a`` / ``stack_b`` are ``(nshot, nsamples)`` on the same grid.  For each
    shot (rows with any non-finite sample skipped) computes the coherence,
    cross-phase, and time-lag cross-correlation.  Returns a dict::

        freq   : (nf,)            frequency axis, Hz  (shared)
        gamma2 : (nshot, nf)      per-shot coherence
        phase  : (nshot, nf)      per-shot cross-phase, radians
        lags   : (nlag,)          lag axis, seconds  (shared)
        xcorr  : (nshot, nlag)    per-shot normalized cross-correlation

    (``nshot`` here is the count of *finite* shots that contributed.)
    """
    stack_a = np.asarray(stack_a, float)
    stack_b = np.asarray(stack_b, float)
    good = _finite_row_mask(stack_a, stack_b)

    freq = lags = None
    g2_rows, ph_rows, xc_rows = [], [], []
    for x, y in zip(stack_a[good], stack_b[good]):
        f, g2 = coherence_spectrum(x, y, dt, nperseg=nperseg)
        _, ph = cross_phase_spectrum(x, y, dt, nperseg=nperseg)
        lag, xc = cross_correlation(x, y, dt)
        if freq is None:
            freq, lags = f, lag
        g2_rows.append(g2)
        ph_rows.append(ph)
        xc_rows.append(xc)

    if not g2_rows:
        raise ValueError("no finite shot pairs to correlate")

    return {
        "freq": freq,
        "gamma2": np.vstack(g2_rows),
        "phase": np.vstack(ph_rows),
        "lags": lags,
        "xcorr": np.vstack(xc_rows),
    }


def _ensemble_xcorr(stack_a, stack_b, dt):
    """Time-lag cross-correlation of the two shot-averaged traces at one position.

    Averages the finite shots of each stack, then cross-correlates the two mean
    traces (:func:`cross_correlation`).  Returns ``(lags, xcorr)`` -- the lag axis
    (seconds) and normalized cross-correlation.  Shared by :func:`xcorr_averaged`
    and :func:`batch_xcorr` so the ensemble lag is computed identically.
    """
    stack_a = np.asarray(stack_a, float)
    stack_b = np.asarray(stack_b, float)
    good = _finite_row_mask(stack_a, stack_b)
    a_avg = stack_a[good].mean(axis=0)
    b_avg = stack_b[good].mean(axis=0)
    return cross_correlation(a_avg, b_avg, dt)


def xcorr_averaged(stack_a, stack_b, dt, nperseg=NPERSEG):
    """Ensemble-averaged correlation over all shots (for the averaged figure).

    Coherence + cross-phase come from :func:`avg_cross_spectrum` (the spectra are
    averaged across shots before the coherence ratio -- the statistically correct
    ensemble estimate).  The time-lag cross-correlation is computed on the two
    shot-averaged traces.  Returns a dict::

        freq   : (nf,)     frequency axis, Hz
        gamma2 : (nf,)     ensemble coherence
        phase  : (nf,)     ensemble cross-phase, radians
        lags   : (nlag,)   lag axis, seconds
        xcorr  : (nlag,)   cross-correlation of the shot-averaged traces
        n_used : int       number of shots that contributed
    """
    freq, gamma2, phase, n_used = avg_cross_spectrum(
        stack_a, stack_b, dt, nperseg=nperseg)
    lags, xcorr = _ensemble_xcorr(stack_a, stack_b, dt)

    return {
        "freq": freq,
        "gamma2": gamma2,
        "phase": phase,
        "lags": lags,
        "xcorr": xcorr,
        "n_used": int(n_used),
    }


# =========================================================================== #
#  Batch (run after verifying in the notebook) -- one HDF5 file at a time
# =========================================================================== #

# Co-located npz: sits next to the raw HDF5 (like the IV .npz), one file per run.
# A run's npz can hold SEVERAL channel-pairs, each under its own key prefix, so
# more pairs can be added later without a new file. Filename is run-derived only.
OUT_NPZ_SUFFIX = "-xcorr-data.npz"


def xcorr_npz_path(ifn):
    """Co-located npz path for a run: ``<run dir>/<run_num>-xcorr-data.npz``.

    Sits beside the raw HDF5 (same convention as the IV ``.npz``); the name is
    derived from the run number only, so every channel-pair for that run shares
    one file.
    """
    run_num = os.path.basename(ifn).split("-")[0]
    return os.path.join(os.path.dirname(ifn), f"{run_num}{OUT_NPZ_SUFFIX}")


def _pair_key(ch_a, ch_b):
    """Key prefix for a channel pair, e.g. ``'machscope-C3__machscope-C4'``.

    Lets one run's npz hold several pairs side by side (``<key>__gamma2`` etc.).
    """
    return f"{ch_a[0]}-{ch_a[1]}__{ch_b[0]}-{ch_b[1]}"


def _position_xy(pos_array, npos, nshot):
    """(x, y) of each of the ``npos`` positions: the first shot of each block.

    Kept with the spectra so a plane map has real axes (not just a position
    index). Shared by :func:`batch_xcorr` and :func:`batch_xcorr_band`.
    """
    return pos_array["x"][::nshot][:npos], pos_array["y"][::nshot][:npos]


def _merge_save_npz(out_path, new_arrays):
    """Merge ``new_arrays`` into the run's co-located npz and rewrite it.

    Loads any existing arrays at ``out_path`` (so other channel pairs already
    stored are kept) and overwrites/adds the keys in ``new_arrays``. Shared by the
    two batch drivers so several pairs -- and both the full-spectrum and band
    entries -- accumulate in one file.
    """
    arrays = {}
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            arrays = {k: d[k] for k in d.files}
    arrays.update(new_arrays)
    np.savez(out_path, **arrays)


def batch_xcorr(ifn, ch_a=CH_A, ch_b=CH_B, tmin_ms=TMIN_MS, tmax_ms=TMAX_MS,
                nperseg=NPERSEG):
    """Per-position ensemble coherence/cross-phase/lag for ONE run -> co-located npz.

    Reads each probe position's ``nshot`` shots for ``ch_a`` / ``ch_b`` and
    incoherently averages the Welch cross/auto spectra over **only that position's
    shots** (the tqdm bar ticks per shot), then forms that position's ensemble
    coherence ``gamma2`` and cross-phase ``phase``, plus the time-lag
    cross-correlation of the two shot-averaged traces (:func:`_ensemble_xcorr`).
    The spatial dimension is kept: results are stored per position, indexed the
    same way as ``positions_array`` (position ``p`` is ``pos_array[p*nshot]``), so
    a plane run can be drawn as an xy map.

    Writes into the run's co-located npz (:func:`xcorr_npz_path`), keyed by the
    channel pair (:func:`_pair_key`): ``freq`` and ``lags`` (shared axes),
    ``pos_x`` / ``pos_y`` (the (x, y) of each of the ``npos`` positions),
    ``<pair>__gamma2`` ``(npos, nf)``, ``<pair>__phase`` ``(npos, nf)``,
    ``<pair>__xcorr`` ``(npos, nlag)`` (the full per-position lag trace),
    ``<pair>__nshots`` ``(npos,)``. An existing npz for the run is **merged** (its
    other pairs are kept), so several pairs accumulate in one file. Returns the npz
    path.
    """
    from tqdm import tqdm

    pos_array, _, _, npos, nshot = jiv.read_lp_positions(ifn)
    run = open_lapd(ifn)
    pos_x, pos_y = _position_xy(pos_array, npos, nshot)

    # One ensemble result PER position: average that position's shots, keep the
    # per-position gamma2/phase/xcorr so the spatial (x, y) structure is preserved.
    # The bar ticks per shot as each position is read off disk (the read is the
    # slow part on a plane run, so per-shot ticks give continuous feedback).
    freq = lags = None
    gamma2 = phase = xcorr = None   # (npos, n*), filled as positions complete
    nshots = np.zeros(npos, dtype=int)
    with tqdm(total=npos * nshot, desc="xcorr", unit="shot") as bar:
        for p in range(npos):
            sa, sb, dt = _read_pair_at_position(
                run, ch_a, ch_b, npos, nshot, p, tmin_ms, tmax_ms)
            try:
                f, g2, ph, n_used = avg_cross_spectrum(sa, sb, dt, nperseg=nperseg)
            except ValueError:
                # No finite shot pair at this position: leave its row NaN so one
                # dead position doesn't abort the whole plane.
                bar.update(sa.shape[0])
                continue
            lag, xc = _ensemble_xcorr(sa, sb, dt)
            if gamma2 is None:
                freq, lags = f, lag
                gamma2 = np.full((npos, f.size), np.nan)
                phase = np.full((npos, f.size), np.nan)
                xcorr = np.full((npos, lag.size), np.nan)
            gamma2[p] = g2
            phase[p] = ph
            xcorr[p] = xc
            nshots[p] = n_used
            bar.update(sa.shape[0])

    if freq is None or nshots.sum() == 0:
        raise ValueError("no finite shot pairs to correlate")

    # Merge into the run's co-located npz (keep any other pairs already stored).
    # Guard the shared freq axis first: pairs in one file must share it.
    out_path = xcorr_npz_path(ifn)
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            if "freq" in d.files and not np.array_equal(freq, d["freq"]):
                raise ValueError(f"{out_path}: freq axis differs from stored pairs "
                                 "(different window / sampling rate)")
    key = _pair_key(ch_a, ch_b)
    _merge_save_npz(out_path, {
        "freq": freq, "lags": lags, "pos_x": pos_x, "pos_y": pos_y,
        f"{key}__gamma2": gamma2, f"{key}__phase": phase,
        f"{key}__xcorr": xcorr, f"{key}__nshots": nshots,
    })
    print(f"\nWrote {out_path}: pair '{key}', {npos} positions "
          f"({int(nshots.sum())} shots total), {freq.size} freq bins, "
          f"{lags.size} lags, window {tmin_ms}-{tmax_ms} ms")
    return out_path


def batch_xcorr_band(ifn, ch_a=CH_A, ch_b=CH_B, fband_khz=FBAND_KHZ,
                     track_peak=False, search_khz=SEARCH_KHZ, deltaf_khz=DELTAF_KHZ,
                     tmin_ms=TMIN_MS, tmax_ms=TMAX_MS, nperseg=NPERSEG):
    """Per-position **narrow-band** scalar coherence + cross-phase -> co-located npz.

    Like :func:`batch_xcorr`, but collapses each position's ensemble spectrum to
    one scalar coherence and one scalar cross-phase (the shot-averaged complex
    spectra are averaged over a band *before* the coherence ratio / phase angle --
    the statistically correct band estimate). Keeps the spatial dimension so a
    plane run becomes a single-frequency coherence map and phase-difference map.

    Two band modes:

    * ``track_peak=False`` (default): a **fixed** band ``fband_khz = (f_lo, f_hi)``
      (kHz) at every position, via :func:`band_cross_spectrum`.
    * ``track_peak=True``: **peak-tracking** for a mode whose frequency drifts
      across the plane -- at each position the coherence peak is located inside
      ``search_khz`` and the band ``peak +- deltaf_khz`` is integrated, via
      :func:`peak_cross_spectrum`. The located peak per position is stored too.

    Writes into the run's co-located npz (:func:`xcorr_npz_path`), keyed by the
    pair (:func:`_pair_key`) with a ``band`` suffix so it coexists with the full
    spectra from :func:`batch_xcorr`: ``pos_x`` / ``pos_y`` (shared),
    ``<pair>__band_gamma2`` ``(npos,)``, ``<pair>__band_phase`` ``(npos,)`` (rad),
    ``<pair>__band_nshots`` ``(npos,)``, ``<pair>__band_fpeak`` ``(npos,)`` (the
    per-position band-center frequency in Hz -- the tracked peak, or the fixed
    band center), and ``<pair>__band_fband`` ``(2,)`` (the search/fixed window in
    Hz). An existing npz is **merged**. Returns the npz path.
    """
    from tqdm import tqdm

    # Select the per-position spectrum function (and its band args) once, then call
    # it uniformly in the loop. peak_cross_spectrum tracks the coherence peak;
    # band_cross_spectrum uses the fixed band. Both return (f_c, gamma2, phase, n).
    if track_peak:
        win = (search_khz[0] * 1e3, search_khz[1] * 1e3)
        spectrum_fn = peak_cross_spectrum
        band_args = (win, deltaf_khz * 1e3)
        desc = f"peak {search_khz[0]:g}-{search_khz[1]:g}kHz +-{deltaf_khz:g}"
    else:
        win = (fband_khz[0] * 1e3, fband_khz[1] * 1e3)
        spectrum_fn = band_cross_spectrum
        band_args = (win,)
        desc = f"band {fband_khz[0]:g}-{fband_khz[1]:g}kHz"

    pos_array, _, _, npos, nshot = jiv.read_lp_positions(ifn)
    run = open_lapd(ifn)
    pos_x, pos_y = _position_xy(pos_array, npos, nshot)

    # One scalar (coherence, phase, band-center) PER position. Dead positions (no
    # finite shot pair) stay NaN so one bad point doesn't abort the whole plane.
    gamma2 = np.full(npos, np.nan)
    phase = np.full(npos, np.nan)
    fpeak = np.full(npos, np.nan)
    nshots = np.zeros(npos, dtype=int)
    with tqdm(total=npos * nshot, desc="xcorr-band", unit="shot") as bar:
        for p in range(npos):
            sa, sb, dt = _read_pair_at_position(
                run, ch_a, ch_b, npos, nshot, p, tmin_ms, tmax_ms)
            try:
                f_c, g2, ph, n_used = spectrum_fn(
                    sa, sb, dt, *band_args, nperseg=nperseg)
            except ValueError:
                bar.update(sa.shape[0])
                continue
            gamma2[p] = g2
            phase[p] = ph
            fpeak[p] = f_c
            nshots[p] = n_used
            bar.update(sa.shape[0])

    if nshots.sum() == 0:
        raise ValueError("no finite shot pairs to correlate")

    out_path = xcorr_npz_path(ifn)
    key = _pair_key(ch_a, ch_b)
    _merge_save_npz(out_path, {
        "pos_x": pos_x, "pos_y": pos_y,
        f"{key}__band_gamma2": gamma2, f"{key}__band_phase": phase,
        f"{key}__band_nshots": nshots, f"{key}__band_fpeak": fpeak,
        f"{key}__band_fband": np.array(win),
    })
    print(f"\nWrote {out_path}: pair '{key}' {desc}, {npos} positions "
          f"({int(nshots.sum())} shots total), window {tmin_ms}-{tmax_ms} ms")
    return out_path


if __name__ == "__main__":

    batch_xcorr(IFN)
