"""Jun-2026 LAPD cross-correlation analysis between two scope channels.

Companion to ``Jun2026_Isat.py``.  Where that module looks at *one* fixed-bias
Isat channel's fluctuation spectrum, this one takes **two** channels and asks how
they relate to each other in the frequency domain:

* **magnitude-squared coherence** ``gamma2(f)`` -- which frequency bands the two
  channels share (0..1), and
* **cross-phase** ``phase(f)`` -- the phase lag per frequency (the standard LAPD
  turbulence diagnostic that, with a known probe separation, gives mode
  direction / speed),

plus a time-lag **cross-correlation** whose peak is the scalar time delay.

Channel identity is a ``(scope, channel)`` pair, e.g. ``("machscope", "C3")`` --
a run has several scope groups, so the scope must be named.  Channel *inspection*
(which scope/channel is what) is not done here; pick the pair yourself.

Two workflows
-------------
1. **Notebook (primary)** -- ``Jun2026_xcorr_explore.ipynb``.  Pick a probe
   position and run the three methods on **all shots at that position**, viewing
   a per-shot figure (shot-to-shot spread) and an ensemble-averaged figure.  The
   per-shot / averaged **analysis** (:func:`xcorr_per_shot` / :func:`xcorr_averaged`)
   lives here; the **figures** live in ``Jun2026_plot`` (as with every other
   Jun-2026 figure), so this module stays pure data/DSP.
2. **Batch (after verifying in the notebook)** -- :func:`batch_xcorr` processes
   **one** run file, averaging the coherence/cross-phase over every shot (the
   progress bar ticks per shot), and writes the result into that run's
   co-located npz (:func:`xcorr_npz_path`), keyed by the channel pair so several
   pairs can share the file.  ``Jun2026_plot`` reloads it to draw a figure.  This
   is the module's ``__main__`` entry.

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
)

import Jun2026_IV as jiv


# --- Configuration (edit-in-place, like Jun2026_Isat.py) --------------------
IFN = r"D:\data\LAPD\jun2026-jia\07-He-800G-bias40V-Isat-p29-plane_2026-06-10.hdf5"
CH_A        = ("machscope", "C2")      # e.g. LP@P29 Isat-L
CH_B        = ("machscope", "C3")      # e.g. LP@P29 Isat-R
TMIN_MS, TMAX_MS = 1.5, 5.0            # analysis time window (ms)
NPERSEG     = 4096                     # Welch segment length (freq res vs variance)


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

    # Time-lag cross-correlation of the shot-averaged traces (finite rows only).
    stack_a = np.asarray(stack_a, float)
    stack_b = np.asarray(stack_b, float)
    good = _finite_row_mask(stack_a, stack_b)
    a_avg = stack_a[good].mean(axis=0)
    b_avg = stack_b[good].mean(axis=0)
    lags, xcorr = cross_correlation(a_avg, b_avg, dt)

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


def batch_xcorr(ifn, ch_a=CH_A, ch_b=CH_B, tmin_ms=TMIN_MS, tmax_ms=TMAX_MS,
                nperseg=NPERSEG):
    """Ensemble coherence/cross-phase for ONE run's channel pair -> co-located npz.

    Reads every position's shots for ``ch_a`` / ``ch_b`` and incoherently averages
    the Welch cross/auto spectra **shot by shot** (the tqdm bar ticks per shot),
    then forms the ensemble coherence ``gamma2`` and cross-phase ``phase``.

    Writes into the run's co-located npz (:func:`xcorr_npz_path`), keyed by the
    channel pair (:func:`_pair_key`): ``freq`` (shared), ``<pair>__gamma2``,
    ``<pair>__phase``, ``<pair>__nshots``. An existing npz for the run is
    **merged** (its other pairs are kept), so several pairs accumulate in one
    file. Returns the npz path.
    """
    from tqdm import tqdm
    from scipy import signal as _sig

    _, _, _, npos, nshot = jiv.read_lp_positions(ifn)
    run = open_lapd(ifn)

    # Accumulate the per-shot Welch spectra as each position is read off disk, so
    # the bar advances continuously (a plane run has thousands of shots and the
    # read itself is the slow part -- pooling first would hang with no feedback).
    kw = None
    freq = None
    pxy_sum = pxx_sum = pyy_sum = 0.0
    n = 0
    with tqdm(total=npos * nshot, desc="xcorr", unit="shot") as bar:
        for p in range(npos):
            sa, sb, dt = _read_pair_at_position(
                run, ch_a, ch_b, npos, nshot, p, tmin_ms, tmax_ms)
            good = _finite_row_mask(sa, sb)
            xs, ys = sa[good], sb[good]
            if kw is None and xs.shape[0]:
                nps = int(min(nperseg, xs.shape[1]))
                kw = dict(fs=1.0 / dt, nperseg=nps, detrend="constant")
            for x, y in zip(xs, ys):
                freq, pxy = _sig.csd(x, y, **kw)
                _, pxx = _sig.welch(x, **kw)
                _, pyy = _sig.welch(y, **kw)
                pxy_sum = pxy_sum + pxy
                pxx_sum = pxx_sum + pxx
                pyy_sum = pyy_sum + pyy
                n += 1
            bar.update(sa.shape[0])

    if n == 0:
        raise ValueError("no finite shot pairs to correlate")

    denom = (pxx_sum * pyy_sum)   # /n cancels in the ratio
    gamma2 = np.where(denom > 0, np.abs(pxy_sum) ** 2 / denom, 0.0)
    phase = np.angle(pxy_sum)

    # Merge into the run's co-located npz (keep any other pairs already stored).
    out_path = xcorr_npz_path(ifn)
    arrays = {}
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            arrays = {k: d[k] for k in d.files}
    if "freq" in arrays and not np.array_equal(freq, arrays["freq"]):
        raise ValueError(f"{out_path}: freq axis differs from stored pairs "
                         "(different window / sampling rate)")
    arrays["freq"] = freq
    key = _pair_key(ch_a, ch_b)
    arrays[f"{key}__gamma2"] = gamma2
    arrays[f"{key}__phase"] = phase
    arrays[f"{key}__nshots"] = n

    np.savez(out_path, **arrays)
    print(f"\nWrote {out_path}: pair '{key}', averaged {n} shots, "
          f"{freq.size} freq bins, window {tmin_ms}-{tmax_ms} ms")
    return out_path


if __name__ == "__main__":

    batch_xcorr(IFN)
